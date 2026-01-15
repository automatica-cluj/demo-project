package com.automatica.fakenews.service;

import com.automatica.fakenews.model.Job;
import com.automatica.fakenews.model.Job.JobStatus;
import com.automatica.fakenews.repository.JobRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

@Service
public class JobService {

    @Autowired
    private JobRepository jobRepository;

    @Autowired
    private JobExecutor jobExecutor;

    @Transactional
    public Job createJob(String jobType, String jobData) {
        Job job = new Job(jobType, jobData);
        job.setStatus(JobStatus.PENDING);
        return jobRepository.save(job);
    }

    public List<Job> getPendingJobs() {
        return jobRepository.findByStatusOrderByCreatedAtDesc(JobStatus.PENDING);
    }

    public List<Job> getApprovedJobs() {
        return jobRepository.findByStatusInOrderByCreatedAtDesc(
            Arrays.asList(JobStatus.APPROVED, JobStatus.EXECUTING)
        );
    }

    public List<Job> getCompletedJobs() {
        return jobRepository.findByStatusInOrderByCreatedAtDesc(
            Arrays.asList(JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.REJECTED)
        );
    }

    public List<Job> getAllJobs() {
        return jobRepository.findAllByOrderByCreatedAtDesc();
    }

    public Optional<Job> getJobById(Long id) {
        return jobRepository.findById(id);
    }

    @Transactional
    public void approveJob(Long id, String approvedBy) {
        Optional<Job> jobOpt = jobRepository.findById(id);
        if (jobOpt.isPresent()) {
            Job job = jobOpt.get();
            if (job.getStatus() == JobStatus.PENDING) {
                job.setStatus(JobStatus.APPROVED);
                job.setApprovedAt(LocalDateTime.now());
                job.setApprovedBy(approvedBy);
                jobRepository.save(job);
                
                // Trigger async execution
                jobExecutor.executeJob(job.getId());
            }
        }
    }

    @Transactional
    public void rejectJob(Long id, String rejectedBy) {
        Optional<Job> jobOpt = jobRepository.findById(id);
        if (jobOpt.isPresent()) {
            Job job = jobOpt.get();
            if (job.getStatus() == JobStatus.PENDING) {
                job.setStatus(JobStatus.REJECTED);
                job.setApprovedAt(LocalDateTime.now());
                job.setApprovedBy(rejectedBy);
                job.setErrorMessage("Rejected by administrator");
                jobRepository.save(job);
            }
        }
    }

    @Transactional
    public void updateJobStatus(Long id, JobStatus status, String result, String error) {
        Optional<Job> jobOpt = jobRepository.findById(id);
        if (jobOpt.isPresent()) {
            Job job = jobOpt.get();
            job.setStatus(status);
            if (status == JobStatus.COMPLETED || status == JobStatus.FAILED) {
                job.setExecutedAt(LocalDateTime.now());
            }
            if (result != null) {
                job.setExecutionResult(result);
            }
            if (error != null) {
                job.setErrorMessage(error);
            }
            jobRepository.save(job);
        }
    }
}
