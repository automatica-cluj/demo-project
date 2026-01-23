package com.automatica.fakenews.model;

import jakarta.persistence.*;
import jakarta.validation.constraints.NotBlank;
import java.time.LocalDateTime;

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

    public FakeNewsReport() {
        this.reportedAt = LocalDateTime.now();
        this.status = ReportStatus.PENDING;
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
}
