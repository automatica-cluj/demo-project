package com.automatica.fakenews.service;

import com.automatica.fakenews.model.Job;
import com.automatica.fakenews.model.Job.JobStatus;
import com.automatica.fakenews.repository.JobRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Optional;

@Service
public class JobExecutor {

    private static final Logger logger = LoggerFactory.getLogger(JobExecutor.class);

    @Autowired
    private JobRepository jobRepository;

    @Async
    @Transactional
    public void executeJob(Long jobId) {
        logger.info("Starting async execution of job {}", jobId);
        
        Optional<Job> jobOpt = jobRepository.findById(jobId);
        if (jobOpt.isEmpty()) {
            logger.error("Job {} not found", jobId);
            return;
        }

        Job job = jobOpt.get();
        
        if (job.getStatus() != JobStatus.APPROVED) {
            logger.error("Job {} is not in APPROVED status, current status: {}", jobId, job.getStatus());
            return;
        }

        try {
            // Update status to EXECUTING
            job.setStatus(JobStatus.EXECUTING);
            jobRepository.save(job);
            logger.info("Job {} status updated to EXECUTING", jobId);

            // Execute the job based on job type
            String result = processJob(job);

            // Update to COMPLETED
            job.setStatus(JobStatus.COMPLETED);
            job.setExecutedAt(LocalDateTime.now());
            job.setExecutionResult(result);
            jobRepository.save(job);
            logger.info("Job {} completed successfully", jobId);

        } catch (Exception e) {
            logger.error("Job {} failed with error: {}", jobId, e.getMessage(), e);
            
            // Update to FAILED
            job.setStatus(JobStatus.FAILED);
            job.setExecutedAt(LocalDateTime.now());
            job.setErrorMessage(e.getMessage());
            jobRepository.save(job);
        }
    }

    private String processJob(Job job) throws Exception {
        logger.info("Processing job type: {}", job.getJobType());
        
        // Simulate processing time
        Thread.sleep(2000);

        switch (job.getJobType()) {
            case "REPORT_VERIFICATION":
                return processReportVerification(job);
            case "DATA_ANALYSIS":
                return processDataAnalysis(job);
            case "NOTIFICATION":
                return processNotification(job);
            default:
                throw new IllegalArgumentException("Unknown job type: " + job.getJobType());
        }
    }

    private String processReportVerification(Job job) {
        logger.info("Verifying report data: {}", job.getJobData());
        // Simulate verification logic
        return "Report verified successfully at " + LocalDateTime.now();
    }

    private String processDataAnalysis(Job job) {
        logger.info("Analyzing data: {}", job.getJobData());
        // Simulate analysis logic
        return "Data analysis completed at " + LocalDateTime.now();
    }

    private String processNotification(Job job) {
        logger.info("Sending notification: {}", job.getJobData());
        // Simulate notification sending
        return "Notification sent at " + LocalDateTime.now();
    }
}
