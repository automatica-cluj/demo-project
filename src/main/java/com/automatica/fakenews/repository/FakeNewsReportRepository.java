package com.automatica.fakenews.repository;

import com.automatica.fakenews.model.FakeNewsReport;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface FakeNewsReportRepository extends JpaRepository<FakeNewsReport, Long> {
    
    List<FakeNewsReport> findByApprovedTrueOrderByApprovedAtDesc();

    List<FakeNewsReport> findByRejectedTrueOrderByRejectedAtDesc();
    
    List<FakeNewsReport> findByApprovedFalseAndRejectedFalseOrderByReportedAtDesc();

    List<FakeNewsReport> findByApprovedTrueOrRejectedTrueOrderByReportedAtDesc();
    
    List<FakeNewsReport> findAllByOrderByReportedAtDesc();
}
