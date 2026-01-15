package com.automatica.fakenews.service;

import com.automatica.fakenews.model.Job;
import com.automatica.fakenews.model.Job.JobStatus;
import com.automatica.fakenews.repository.JobRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class JobServiceTest {

    @Mock
    private JobRepository jobRepository;

    @Mock
    private JobExecutor jobExecutor;

    @InjectMocks
    private JobService jobService;

    @Test
    void testCreateJob_Success() {
        // Given
        String jobType = "REPORT_VERIFICATION";
        String jobData = "Test data";
        Job expectedJob = new Job(jobType, jobData);
        
        when(jobRepository.save(any(Job.class))).thenReturn(expectedJob);

        // When
        Job createdJob = jobService.createJob(jobType, jobData);

        // Then
        assertNotNull(createdJob);
        verify(jobRepository, times(1)).save(any(Job.class));
        
        ArgumentCaptor<Job> jobCaptor = ArgumentCaptor.forClass(Job.class);
        verify(jobRepository).save(jobCaptor.capture());
        Job savedJob = jobCaptor.getValue();
        
        assertEquals(jobType, savedJob.getJobType());
        assertEquals(jobData, savedJob.getJobData());
        assertEquals(JobStatus.PENDING, savedJob.getStatus());
    }

    @Test
    void testApproveJob_SetsStatusAndTriggersExecution() {
        // Given
        Job job = new Job("REPORT_VERIFICATION", "Test data");
        job.setId(1L);
        job.setStatus(JobStatus.PENDING);

        when(jobRepository.findById(1L)).thenReturn(Optional.of(job));
        when(jobRepository.save(any(Job.class))).thenReturn(job);

        // When
        jobService.approveJob(1L, "admin");

        // Then
        ArgumentCaptor<Job> jobCaptor = ArgumentCaptor.forClass(Job.class);
        verify(jobRepository).save(jobCaptor.capture());
        Job savedJob = jobCaptor.getValue();

        assertEquals(JobStatus.APPROVED, savedJob.getStatus());
        assertEquals("admin", savedJob.getApprovedBy());
        assertNotNull(savedJob.getApprovedAt());
        
        // Verify async execution was triggered
        verify(jobExecutor, times(1)).executeJob(1L);
    }

    @Test
    void testApproveJob_JobNotFound_DoesNothing() {
        // Given
        when(jobRepository.findById(999L)).thenReturn(Optional.empty());

        // When
        jobService.approveJob(999L, "admin");

        // Then
        verify(jobRepository, never()).save(any(Job.class));
        verify(jobExecutor, never()).executeJob(anyLong());
    }

    @Test
    void testApproveJob_NonPendingJob_DoesNotApprove() {
        // Given
        Job job = new Job("REPORT_VERIFICATION", "Test data");
        job.setId(1L);
        job.setStatus(JobStatus.COMPLETED);

        when(jobRepository.findById(1L)).thenReturn(Optional.of(job));

        // When
        jobService.approveJob(1L, "admin");

        // Then
        verify(jobRepository, never()).save(any(Job.class));
        verify(jobExecutor, never()).executeJob(anyLong());
    }

    @Test
    void testRejectJob_SetsStatusAndReason() {
        // Given
        Job job = new Job("REPORT_VERIFICATION", "Test data");
        job.setId(1L);
        job.setStatus(JobStatus.PENDING);

        when(jobRepository.findById(1L)).thenReturn(Optional.of(job));
        when(jobRepository.save(any(Job.class))).thenReturn(job);

        // When
        jobService.rejectJob(1L, "admin");

        // Then
        ArgumentCaptor<Job> jobCaptor = ArgumentCaptor.forClass(Job.class);
        verify(jobRepository).save(jobCaptor.capture());
        Job savedJob = jobCaptor.getValue();

        assertEquals(JobStatus.REJECTED, savedJob.getStatus());
        assertEquals("admin", savedJob.getApprovedBy());
        assertNotNull(savedJob.getApprovedAt());
        assertEquals("Rejected by administrator", savedJob.getErrorMessage());
    }

    @Test
    void testGetPendingJobs_DelegatesToRepository() {
        // Given
        Job job1 = new Job("TYPE1", "data1");
        Job job2 = new Job("TYPE2", "data2");
        List<Job> pendingJobs = Arrays.asList(job1, job2);
        
        when(jobRepository.findByStatusOrderByCreatedAtDesc(JobStatus.PENDING))
            .thenReturn(pendingJobs);

        // When
        List<Job> result = jobService.getPendingJobs();

        // Then
        assertNotNull(result);
        assertEquals(2, result.size());
        verify(jobRepository, times(1)).findByStatusOrderByCreatedAtDesc(JobStatus.PENDING);
    }

    @Test
    void testGetApprovedJobs_DelegatesToRepository() {
        // Given
        Job job1 = new Job("TYPE1", "data1");
        job1.setStatus(JobStatus.APPROVED);
        Job job2 = new Job("TYPE2", "data2");
        job2.setStatus(JobStatus.EXECUTING);
        List<Job> approvedJobs = Arrays.asList(job1, job2);
        
        when(jobRepository.findByStatusInOrderByCreatedAtDesc(
            Arrays.asList(JobStatus.APPROVED, JobStatus.EXECUTING)))
            .thenReturn(approvedJobs);

        // When
        List<Job> result = jobService.getApprovedJobs();

        // Then
        assertNotNull(result);
        assertEquals(2, result.size());
    }

    @Test
    void testUpdateJobStatus_UpdatesAllFields() {
        // Given
        Job job = new Job("REPORT_VERIFICATION", "Test data");
        job.setId(1L);
        job.setStatus(JobStatus.EXECUTING);

        when(jobRepository.findById(1L)).thenReturn(Optional.of(job));
        when(jobRepository.save(any(Job.class))).thenReturn(job);

        // When
        jobService.updateJobStatus(1L, JobStatus.COMPLETED, "Success result", null);

        // Then
        ArgumentCaptor<Job> jobCaptor = ArgumentCaptor.forClass(Job.class);
        verify(jobRepository).save(jobCaptor.capture());
        Job savedJob = jobCaptor.getValue();

        assertEquals(JobStatus.COMPLETED, savedJob.getStatus());
        assertEquals("Success result", savedJob.getExecutionResult());
        assertNotNull(savedJob.getExecutedAt());
    }

    @Test
    void testUpdateJobStatus_WithError() {
        // Given
        Job job = new Job("REPORT_VERIFICATION", "Test data");
        job.setId(1L);
        job.setStatus(JobStatus.EXECUTING);

        when(jobRepository.findById(1L)).thenReturn(Optional.of(job));
        when(jobRepository.save(any(Job.class))).thenReturn(job);

        // When
        jobService.updateJobStatus(1L, JobStatus.FAILED, null, "Error message");

        // Then
        ArgumentCaptor<Job> jobCaptor = ArgumentCaptor.forClass(Job.class);
        verify(jobRepository).save(jobCaptor.capture());
        Job savedJob = jobCaptor.getValue();

        assertEquals(JobStatus.FAILED, savedJob.getStatus());
        assertEquals("Error message", savedJob.getErrorMessage());
        assertNotNull(savedJob.getExecutedAt());
    }
}
