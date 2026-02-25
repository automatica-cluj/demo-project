package com.automatica.fakenews.model;

import jakarta.persistence.*;
import jakarta.validation.constraints.NotBlank;
import java.time.LocalDateTime;
import java.util.Locale;

@Entity
@Table(name = "fake_news_reports")
public class FakeNewsReport {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotBlank(message = "News source is required")
    @Column(nullable = false)
    private String newsSource;

    @NotBlank(message = "URL is required")
    @Column(nullable = false)
    private String url;

    @NotBlank(message = "Category is required")
    @Column(nullable = false)
    private String category;

    @Column(columnDefinition = "TEXT")
    private String description;

    @Column(nullable = false)
    private LocalDateTime reportedAt;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private ReportStatus status;

    @Column
    private LocalDateTime processedAt;

    @Column
    private String processedBy;

    @Column
    private String detectionResult;

    @Column
    private Double detectionScore;

    @Column
    private String detectionProvider;

    @Column(columnDefinition = "TEXT")
    private String detectionDetails;

    public FakeNewsReport() {
        this.reportedAt = LocalDateTime.now();
        this.status = ReportStatus.PENDING;
    }

    public String getDetectionResult() {
        return detectionResult;
    }

    public void setDetectionResult(String detectionResult) {
        this.detectionResult = detectionResult;
    }

    public Double getDetectionScore() {
        return detectionScore;
    }

    public void setDetectionScore(Double detectionScore) {
        this.detectionScore = detectionScore;
    }

    public String getDetectionProvider() {
        return detectionProvider;
    }

    public void setDetectionProvider(String detectionProvider) {
        this.detectionProvider = detectionProvider;
    }

    public String getDetectionDetails() {
        return detectionDetails;
    }

    public void setDetectionDetails(String detectionDetails) {
        this.detectionDetails = detectionDetails;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getNewsSource() {
        return newsSource;
    }

    public void setNewsSource(String newsSource) {
        this.newsSource = newsSource;
    }

    public String getUrl() {
        return url;
    }

    public void setUrl(String url) {
        this.url = url;
    }

    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public LocalDateTime getReportedAt() {
        return reportedAt;
    }

    public void setReportedAt(LocalDateTime reportedAt) {
        this.reportedAt = reportedAt;
    }

    public ReportStatus getStatus() {
        return status;
    }

    public void setStatus(ReportStatus status) {
        this.status = status;
    }

    public LocalDateTime getProcessedAt() {
        return processedAt;
    }

    public void setProcessedAt(LocalDateTime processedAt) {
        this.processedAt = processedAt;
    }

    public String getProcessedBy() {
        return processedBy;
    }

    public void setProcessedBy(String processedBy) {
        this.processedBy = processedBy;
    }

    public boolean isApproved() {
        return status == ReportStatus.APPROVED;
    }

    public boolean isRejected() {
        return status == ReportStatus.REJECTED;
    }

    public boolean isPending() {
        return status == ReportStatus.PENDING;
    }

    public boolean isInProgress() {
        return status == ReportStatus.IN_PROGRESS;
    }

    @Transient
    public double getFakeProbability() {
        if (detectionScore == null || detectionResult == null) {
            return 0.0d;
        }
        return isFakeLabel() ? detectionScore : 1.0d - detectionScore;
    }

    @Transient
    public double getRealProbability() {
        if (detectionScore == null || detectionResult == null) {
            return 0.0d;
        }
        return isRealLabel() ? detectionScore : 1.0d - detectionScore;
    }

    @Transient
    public double getAiMargin() {
        if (detectionScore == null || detectionResult == null) {
            return 0.0d;
        }
        return Math.abs(getFakeProbability() - getRealProbability());
    }

    @Transient
    public String getAiRecommendation() {
        if (detectionScore == null || detectionResult == null) {
            return "NO_DATA";
        }
        return getFakeProbability() >= getRealProbability() ? "REJECT" : "APPROVE";
    }

    private boolean isFakeLabel() {
        String normalized = detectionResult.trim().toUpperCase(Locale.ROOT);
        return normalized.equals("FAKE") || normalized.equals("LABEL_0");
    }

    private boolean isRealLabel() {
        String normalized = detectionResult.trim().toUpperCase(Locale.ROOT);
        return normalized.equals("REAL") || normalized.equals("TRUE") || normalized.equals("LABEL_1");
    }
}
