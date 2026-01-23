package com.automatica.fakenews.repository;

import com.automatica.fakenews.model.FakeNewsReport;
import com.automatica.fakenews.model.ReportStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface FakeNewsReportRepository extends JpaRepository<FakeNewsReport, Long> {

    List<FakeNewsReport> findByStatusOrderByProcessedAtDesc(ReportStatus status);

    List<FakeNewsReport> findByStatusOrderByReportedAtDesc(ReportStatus status);

    @Query("SELECT r FROM FakeNewsReport r WHERE r.status = com.automatica.fakenews.model.ReportStatus.APPROVED OR r.status = com.automatica.fakenews.model.ReportStatus.REJECTED ORDER BY r.processedAt DESC")
    List<FakeNewsReport> findApprovedAndRejectedReportsOrderByProcessedAtDesc();

    List<FakeNewsReport> findAllByOrderByReportedAtDesc();
}
