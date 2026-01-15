package com.automatica.fakenews.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "jobs")
public class Job {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String jobType;

    @Column(columnDefinition = "TEXT")
    private String jobData;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private JobStatus status = JobStatus.PENDING;

    @Column(nullable = false)
    private LocalDateTime createdAt;

    @Column
    private LocalDateTime approvedAt;

    @Column
    private String approvedBy;

    @Column
    private LocalDateTime executedAt;

    @Column(columnDefinition = "TEXT")
    private String executionResult;

    @Column(columnDefinition = "TEXT")
    private String errorMessage;

    public Job() {
        this.createdAt = LocalDateTime.now();
    }

    public Job(String jobType, String jobData) {
        this();
        this.jobType = jobType;
        this.jobData = jobData;
    }

    public enum JobStatus {
        PENDING,      // Job created, waiting for approval
        APPROVED,     // Job approved by admin, ready for execution
        EXECUTING,    // Job is currently being executed
        COMPLETED,    // Job executed successfully
        FAILED,       // Job execution failed
        REJECTED      // Job rejected by admin
    }

    // Getters and Setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getJobType() {
        return jobType;
    }

    public void setJobType(String jobType) {
        this.jobType = jobType;
    }

    public String getJobData() {
        return jobData;
    }

    public void setJobData(String jobData) {
        this.jobData = jobData;
    }

    public JobStatus getStatus() {
        return status;
    }

    public void setStatus(JobStatus status) {
        this.status = status;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public LocalDateTime getApprovedAt() {
        return approvedAt;
    }

    public void setApprovedAt(LocalDateTime approvedAt) {
        this.approvedAt = approvedAt;
    }

    public String getApprovedBy() {
        return approvedBy;
    }

    public void setApprovedBy(String approvedBy) {
        this.approvedBy = approvedBy;
    }

    public LocalDateTime getExecutedAt() {
        return executedAt;
    }

    public void setExecutedAt(LocalDateTime executedAt) {
        this.executedAt = executedAt;
    }

    public String getExecutionResult() {
        return executionResult;
    }

    public void setExecutionResult(String executionResult) {
        this.executionResult = executionResult;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }
}
