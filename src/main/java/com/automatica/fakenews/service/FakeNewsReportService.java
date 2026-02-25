package com.automatica.fakenews.service;

import com.automatica.fakenews.dto.huggingface.LabelScore;
import com.fasterxml.jackson.databind.JsonNode;
import com.automatica.fakenews.model.FakeNewsReport;
import com.automatica.fakenews.model.ReportStatus;
import com.automatica.fakenews.repository.FakeNewsReportRepository;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.StringUtils;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.client.RestTemplate;

import java.net.URI;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.TimeUnit;

@Service
public class FakeNewsReportService {

    private static final Logger logger = LoggerFactory.getLogger(FakeNewsReportService.class);
    private static final int MAX_FETCHED_TEXT_LENGTH = 10_000;
    private static final int MAX_ANALYSIS_TEXT_LENGTH = 5_000;
    private static final int MAX_ANALYSIS_WORDS = 300;
    private static final int MAX_HF_RETRIES = 3;

    private final FakeNewsReportRepository reportRepository;
    private final RestTemplate huggingFaceRestTemplate;
    private final RestTemplate contentFetchRestTemplate;

    @Value("${huggingface.api.url}")
    private String huggingFaceApiUrl;

    @Value("${huggingface.api.token}")
    private String huggingFaceApiToken;

    @Value("${huggingface.api.local-token:}")
    private String localHuggingFaceApiToken;

    @Value("${openai.api.url:https://api.openai.com/v1/chat/completions}")
    private String openAiApiUrl;

    @Value("${openai.api.token:}")
    private String openAiApiToken;

    @Value("${openai.api.model:gpt-4o-mini}")
    private String openAiModel;

    public FakeNewsReportService(FakeNewsReportRepository reportRepository, RestTemplateBuilder restTemplateBuilder) {
        this.reportRepository = reportRepository;
        this.huggingFaceRestTemplate = restTemplateBuilder
                .setConnectTimeout(Duration.ofSeconds(5))
                .setReadTimeout(Duration.ofSeconds(25))
                .build();
        this.contentFetchRestTemplate = restTemplateBuilder
                .setConnectTimeout(Duration.ofSeconds(2))
                .setReadTimeout(Duration.ofSeconds(4))
                .build();
    }

    @PostConstruct
    void logHuggingFaceConfiguration() {
        logger.info("Hugging Face URL: {}", huggingFaceApiUrl);
        logger.info("Hugging Face token configured: {}", StringUtils.hasText(resolveHuggingFaceToken()));
        logger.info("OpenAI fallback configured: {}", StringUtils.hasText(resolveOpenAiToken()));
    }

    @Transactional
    public void analyzeAndSetReportDetection(FakeNewsReport report) {
        String textToAnalyze = buildTextToAnalyze(report);
        if (!StringUtils.hasText(textToAnalyze)) {
            setDetectionOutcome(report, "REAL", 0.50d, "SYSTEM_NEUTRAL", "No analyzable content");
            logger.info("Report {} has no analyzable text, using neutral fallback.", report.getId());
            return;
        }

        Optional<AnalysisOutcome> huggingFaceOutcome = analyzeWithHuggingFace(report, textToAnalyze);
        if (huggingFaceOutcome.isPresent()) {
            applyOutcome(report, huggingFaceOutcome.get());
            return;
        }

        Optional<AnalysisOutcome> openAiOutcome = analyzeWithOpenAi(report, textToAnalyze);
        if (openAiOutcome.isPresent()) {
            applyOutcome(report, openAiOutcome.get());
            return;
        }

        logger.info("Using deterministic heuristic fallback for report {} after HF and OpenAI fallback were unavailable.",
                report.getId());
        applyHeuristicFallback(report, textToAnalyze, "hf_and_openai_unavailable");
    }

    private Optional<AnalysisOutcome> analyzeWithHuggingFace(FakeNewsReport report, String textToAnalyze) {
        String effectiveToken = resolveHuggingFaceToken();
        if (!StringUtils.hasText(effectiveToken)) {
            logger.info("Hugging Face token is not configured for report {}.", report.getId());
            return Optional.empty();
        }

        try {
            for (int attempt = 1; attempt <= MAX_HF_RETRIES; attempt++) {
                HttpHeaders headers = new HttpHeaders();
                headers.setContentType(MediaType.APPLICATION_JSON);
                headers.setAccept(List.of(MediaType.APPLICATION_JSON));
                headers.set("Authorization", "Bearer " + effectiveToken);

                HttpEntity<Object> entity = new HttpEntity<>(
                        Map.of(
                                "inputs", textToAnalyze,
                                "options", Map.of("wait_for_model", true),
                                "parameters", Map.of(
                                        "truncation", true,
                                        "max_length", 512
                                )
                        ),
                        headers
                );

                ResponseEntity<JsonNode> response = huggingFaceRestTemplate.exchange(
                        huggingFaceApiUrl,
                        HttpMethod.POST,
                        entity,
                        JsonNode.class
                );

                JsonNode body = response.getBody();
                if (isModelLoadingResponse(body)) {
                    long waitSeconds = estimateWaitSeconds(body, attempt);
                    logger.info("Hugging Face model loading for report {}. Retry {}/{} in {}s.",
                            report.getId(), attempt, MAX_HF_RETRIES, waitSeconds);
                    sleepQuietly(waitSeconds);
                    continue;
                }

                List<LabelScore> scores = extractScores(body);
                Optional<LabelScore> maxScore = scores.stream().max(Comparator.comparing(LabelScore::getScore));
                if (maxScore.isPresent()) {
                    String normalizedLabel = normalizeLabel(maxScore.get().getLabel());
                    double normalizedScore = normalizeScore(maxScore.get().getScore());
                    if (normalizedLabel == null) {
                        logger.warn("Hugging Face returned unknown label for report {}: {}", report.getId(), maxScore.get().getLabel());
                        return Optional.empty();
                    }
                    return Optional.of(new AnalysisOutcome(
                            normalizedLabel,
                            normalizedScore,
                            "HUGGING_FACE",
                            "Provider: Hugging Face model response"
                    ));
                }
            }
        } catch (RestClientResponseException e) {
            String responseBody = e.getResponseBodyAsString();
            if (responseBody != null && responseBody.length() > 500) {
                responseBody = responseBody.substring(0, 500);
            }
            logger.error("Hugging Face HTTP error for report {}: status={}, body={}",
                    report.getId(), e.getRawStatusCode(), responseBody);
        } catch (Exception e) {
            logger.error("Error analyzing report {} with Hugging Face API: {}", report.getId(), e.getMessage(), e);
        }

        return Optional.empty();
    }

    private Optional<AnalysisOutcome> analyzeWithOpenAi(FakeNewsReport report, String textToAnalyze) {
        String token = resolveOpenAiToken();
        if (!StringUtils.hasText(token)) {
            return Optional.empty();
        }

        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.setAccept(List.of(MediaType.APPLICATION_JSON));
            headers.setBearerAuth(token);

            String systemPrompt = "You classify news credibility. Return only JSON with keys: label (FAKE or REAL), score (0..1), reason.";
            String userPrompt = "Classify this report text:\n" + textToAnalyze;

            Map<String, Object> requestBody = Map.of(
                    "model", openAiModel,
                    "messages", List.of(
                            Map.of("role", "system", "content", systemPrompt),
                            Map.of("role", "user", "content", userPrompt)
                    ),
                    "temperature", 0,
                    "response_format", Map.of("type", "json_object")
            );

            HttpEntity<Object> entity = new HttpEntity<>(requestBody, headers);
            ResponseEntity<JsonNode> response = huggingFaceRestTemplate.exchange(
                    openAiApiUrl,
                    HttpMethod.POST,
                    entity,
                    JsonNode.class
            );

            JsonNode contentNode = response.getBody()
                    .path("choices")
                    .path(0)
                    .path("message")
                    .path("content");

            if (!contentNode.isTextual()) {
                return Optional.empty();
            }

            JsonNode jsonContent = com.fasterxml.jackson.databind.json.JsonMapper.builder()
                    .build()
                    .readTree(contentNode.asText());

            String label = normalizeLabel(jsonContent.path("label").asText(null));
            if (label == null) {
                return Optional.empty();
            }

            double score = normalizeScore(jsonContent.path("score").asDouble(0.5d));
            String reason = jsonContent.path("reason").asText("Provider: OpenAI fallback");
            return Optional.of(new AnalysisOutcome(label, score, "OPENAI_FALLBACK", reason));
        } catch (Exception e) {
            logger.warn("OpenAI fallback failed for report {}: {}", report.getId(), e.getMessage());
            return Optional.empty();
        }
    }

    private String resolveHuggingFaceToken() {
        if (StringUtils.hasText(localHuggingFaceApiToken)) {
            return localHuggingFaceApiToken.trim();
        }
        if (StringUtils.hasText(huggingFaceApiToken)) {
            return huggingFaceApiToken.trim();
        }
        return null;
    }

    private String resolveOpenAiToken() {
        if (StringUtils.hasText(openAiApiToken)) {
            return openAiApiToken.trim();
        }
        return null;
    }

    private boolean isModelLoadingResponse(JsonNode body) {
        if (body == null || !body.isObject()) {
            return false;
        }
        if (!body.has("error")) {
            return false;
        }
        String error = body.path("error").asText("").toLowerCase(Locale.ROOT);
        return error.contains("loading");
    }

    private long estimateWaitSeconds(JsonNode body, int attempt) {
        double estimated = body.path("estimated_time").asDouble(0d);
        if (estimated > 0d) {
            return Math.max(1L, Math.min(8L, (long) Math.ceil(estimated)));
        }
        return Math.min(2L * attempt, 6L);
    }

    private void sleepQuietly(long seconds) {
        try {
            TimeUnit.SECONDS.sleep(seconds);
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        }
    }

    private List<LabelScore> extractScores(JsonNode body) {
        if (body == null || body.isNull()) {
            return List.of();
        }

        if (body.isObject() && body.has("error")) {
            String errorMessage = body.get("error").asText();
            if (errorMessage.toLowerCase(Locale.ROOT).contains("loading")) {
                logger.info("Hugging Face model is loading: {}", errorMessage);
            } else {
                logger.warn("Hugging Face API returned error: {}", errorMessage);
            }
            return List.of();
        }

        JsonNode candidateArray = body;
        if (body.isArray() && !body.isEmpty() && body.get(0).isArray()) {
            candidateArray = body.get(0);
        }
        if (!candidateArray.isArray()) {
            return List.of();
        }

        List<LabelScore> scores = new ArrayList<>();
        for (JsonNode node : candidateArray) {
            if (node.hasNonNull("label") && node.hasNonNull("score")) {
                if (!node.get("score").isNumber()) {
                    continue;
                }
                LabelScore score = new LabelScore();
                score.setLabel(node.get("label").asText());
                score.setScore(node.get("score").asDouble());
                scores.add(score);
            }
        }
        return scores;
    }

    private String buildTextToAnalyze(FakeNewsReport report) {
        StringBuilder text = new StringBuilder();
        appendIfPresent(text, report.getNewsSource());
        appendIfPresent(text, report.getCategory());
        appendIfPresent(text, report.getDescription());

        String extractedArticleText = fetchAndExtractArticleText(report.getUrl());
        if (StringUtils.hasText(extractedArticleText)) {
            appendIfPresent(text, extractedArticleText);
        } else {
            appendIfPresent(text, report.getUrl());
        }

        String combinedText = text.toString().trim();
        combinedText = truncateToWords(combinedText, MAX_ANALYSIS_WORDS);
        if (combinedText.length() > MAX_ANALYSIS_TEXT_LENGTH) {
            return combinedText.substring(0, MAX_ANALYSIS_TEXT_LENGTH);
        }
        return combinedText;
    }

    private String truncateToWords(String text, int maxWords) {
        if (!StringUtils.hasText(text) || maxWords <= 0) {
            return text;
        }
        String[] words = text.trim().split("\\s+");
        if (words.length <= maxWords) {
            return text;
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < maxWords; i++) {
            if (i > 0) {
                sb.append(' ');
            }
            sb.append(words[i]);
        }
        return sb.toString();
    }

    private String fetchAndExtractArticleText(String url) {
        if (!StringUtils.hasText(url)) {
            return null;
        }

        try {
            URI uri = URI.create(url.trim());
            String scheme = uri.getScheme();
            if (scheme == null || !(scheme.equalsIgnoreCase("http") || scheme.equalsIgnoreCase("https"))) {
                logger.info("Skipping non-http(s) URL for content extraction: {}", url);
                return null;
            }

            HttpHeaders headers = new HttpHeaders();
            headers.setAccept(List.of(MediaType.TEXT_HTML, MediaType.ALL));
            headers.set("User-Agent", "FakeNewsReporter/1.0");
            HttpEntity<Void> entity = new HttpEntity<>(headers);

            ResponseEntity<String> response = contentFetchRestTemplate.exchange(uri, HttpMethod.GET, entity, String.class);
            String html = response.getBody();
            if (!StringUtils.hasText(html)) {
                return null;
            }

            String extractedText = stripHtmlToText(html);
            if (!StringUtils.hasText(extractedText)) {
                return null;
            }

            if (extractedText.length() > MAX_FETCHED_TEXT_LENGTH) {
                return extractedText.substring(0, MAX_FETCHED_TEXT_LENGTH);
            }
            return extractedText;
        } catch (Exception e) {
            logger.warn("Could not extract text from URL {}: {}", url, e.getMessage());
            return null;
        }
    }

    private String stripHtmlToText(String html) {
        String withoutScripts = html.replaceAll("(?is)<script[^>]*>.*?</script>", " ");
        String withoutStyles = withoutScripts.replaceAll("(?is)<style[^>]*>.*?</style>", " ");
        String withoutTags = withoutStyles.replaceAll("(?is)<[^>]+>", " ");
        String normalizedWhitespace = withoutTags.replaceAll("\\s+", " ").trim();
        return decodeBasicHtmlEntities(normalizedWhitespace);
    }

    private String decodeBasicHtmlEntities(String text) {
        return text
                .replace("&nbsp;", " ")
                .replace("&amp;", "&")
                .replace("&quot;", "\"")
                .replace("&#39;", "'")
                .replace("&lt;", "<")
                .replace("&gt;", ">");
    }

    private void appendIfPresent(StringBuilder target, String value) {
        if (StringUtils.hasText(value)) {
            if (!target.isEmpty()) {
                target.append(' ');
            }
            target.append(value.trim());
        }
    }

    private String normalizeLabel(String rawLabel) {
        if (rawLabel == null) {
            return null;
        }
        String normalized = rawLabel.trim().toUpperCase(Locale.ROOT);
        if ("LABEL_0".equals(normalized) || normalized.contains("FAKE")) {
            return "FAKE";
        }
        if ("LABEL_1".equals(normalized) || normalized.contains("REAL") || normalized.contains("TRUE")) {
            return "REAL";
        }
        return null;
    }

    private double normalizeScore(double score) {
        if (Double.isNaN(score) || Double.isInfinite(score)) {
            return 0.5d;
        }
        return Math.max(0.0d, Math.min(1.0d, score));
    }

    private void applyOutcome(FakeNewsReport report, AnalysisOutcome outcome) {
        setDetectionOutcome(report, outcome.label(), outcome.score(), outcome.provider(), outcome.details());
        logger.info("Report {} analyzed via {}. Result: {}, Score: {}",
                report.getId(), outcome.provider(), outcome.label(), outcome.score());
    }

    private void setDetectionOutcome(FakeNewsReport report, String result, double score, String provider, String details) {
        report.setDetectionResult(result);
        report.setDetectionScore(normalizeScore(score));
        report.setDetectionProvider(provider);
        report.setDetectionDetails(details);
    }

    private void applyHeuristicFallback(FakeNewsReport report, String text, String reason) {
        String normalizedText = text.toLowerCase(Locale.ROOT);
        Set<String> fakeSignals = Set.of(
                "breaking", "shocking", "secret", "conspiracy", "miracle cure",
                "they don't want you to know", "100% proof", "exposed", "hoax",
                "scam", "unbelievable", "urgent", "viral", "rumor", "alleged"
        );
        Set<String> realSignals = Set.of(
                "according to", "official statement", "report by", "data shows",
                "sources said", "analysis", "study", "evidence", "reuters", "associated press"
        );

        int fakeHits = countSignalHits(normalizedText, fakeSignals);
        int realHits = countSignalHits(normalizedText, realSignals);

        String result;
        double confidence;
        if (fakeHits >= realHits) {
            result = "FAKE";
            confidence = Math.min(0.55d + (fakeHits - realHits) * 0.08d + fakeHits * 0.02d, 0.85d);
        } else {
            result = "REAL";
            confidence = Math.min(0.55d + (realHits - fakeHits) * 0.08d + realHits * 0.02d, 0.85d);
        }

        setDetectionOutcome(report, result, Math.max(0.51d, confidence), "HEURISTIC_FALLBACK",
                "Fallback reason: " + reason);
        logger.info("Fallback analysis used for report {} (reason={}) -> {} ({})",
                report.getId(), reason, report.getDetectionResult(), report.getDetectionScore());
    }

    private int countSignalHits(String text, Set<String> signals) {
        int count = 0;
        for (String signal : signals) {
            if (text.contains(signal)) {
                count++;
            }
        }
        return count;
    }

    private record AnalysisOutcome(String label, double score, String provider, String details) {
    }

    public List<FakeNewsReport> getApprovedReports() {
        return reportRepository.findByStatusOrderByProcessedAtDesc(ReportStatus.APPROVED);
    }

    public List<FakeNewsReport> getPendingReports() {
        return reportRepository.findByStatusOrderByReportedAtDesc(ReportStatus.PENDING);
    }

    public List<FakeNewsReport> getInProgressReports() {
        return reportRepository.findByStatusOrderByReportedAtDesc(ReportStatus.IN_PROGRESS);
    }

    public List<FakeNewsReport> getAllReports() {
        return reportRepository.findAllByOrderByReportedAtDesc();
    }

    public Optional<FakeNewsReport> getReportById(Long id) {
        return reportRepository.findById(id);
    }

    @Transactional
    public FakeNewsReport saveReport(FakeNewsReport report) {
        return reportRepository.save(report);
    }

    @Transactional
    public void setReportStatus(Long id, ReportStatus status, String processedBy) {
        Optional<FakeNewsReport> reportOpt = reportRepository.findById(id);
        if (reportOpt.isPresent()) {
            FakeNewsReport report = reportOpt.get();
            if (report.getStatus() == ReportStatus.REJECTED) {
                logger.info("Ignoring status change for rejected report {}", id);
                return;
            }
            report.setStatus(status);
            report.setProcessedAt(LocalDateTime.now());
            report.setProcessedBy(processedBy);
            reportRepository.save(report);
        }
    }

    @Transactional
    public void setInProgressReport(Long id) {
        Optional<FakeNewsReport> reportOpt = reportRepository.findById(id);
        if (reportOpt.isPresent()) {
            FakeNewsReport report = reportOpt.get();
            report.setStatus(ReportStatus.IN_PROGRESS);
            reportRepository.save(report);
        }
    }

    public List<FakeNewsReport> getRejectedReports() {
        return reportRepository.findByStatusOrderByProcessedAtDesc(ReportStatus.REJECTED);
    }

    public List<FakeNewsReport> getPublicReports() {
        return reportRepository.findApprovedAndRejectedReportsOrderByProcessedAtDesc();
    }

    @Transactional
    public void deleteReport(Long id) {
        Optional<FakeNewsReport> reportOpt = reportRepository.findById(id);
        if (reportOpt.isPresent() && reportOpt.get().getStatus() == ReportStatus.REJECTED) {
            logger.info("Ignoring delete request for rejected report {}", id);
            return;
        }
        reportRepository.deleteById(id);
    }

    @Transactional
    public void enrichReportsWithAiDetection(List<FakeNewsReport> reports) {
        for (FakeNewsReport report : reports) {
            if (report.getDetectionResult() != null
                    && report.getDetectionScore() != null
                    && report.getDetectionProvider() == null) {
                report.setDetectionProvider("LEGACY_RESULT");
                report.setDetectionDetails("Result generated before provider tracking was introduced.");
                reportRepository.save(report);
                continue;
            }

            if (report.getDetectionResult() == null || report.getDetectionScore() == null) {
                analyzeAndSetReportDetection(report);
                if (report.getDetectionResult() != null && report.getDetectionScore() != null) {
                    reportRepository.save(report);
                }
            }
        }
    }
}
