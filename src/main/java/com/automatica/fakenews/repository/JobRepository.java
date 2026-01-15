package com.automatica.fakenews.repository;

import com.automatica.fakenews.model.Job;
import com.automatica.fakenews.model.Job.JobStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface JobRepository extends JpaRepository<Job, Long> {
    
    List<Job> findByStatusOrderByCreatedAtDesc(JobStatus status);
    
    List<Job> findByStatusInOrderByCreatedAtDesc(List<JobStatus> statuses);
    
    List<Job> findAllByOrderByCreatedAtDesc();
}
