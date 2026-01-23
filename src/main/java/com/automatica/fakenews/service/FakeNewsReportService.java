package com.automatica.fakenews.service;

import com.automatica.fakenews.model.FakeNewsReport;
import com.automatica.fakenews.model.ReportStatus;
import com.automatica.fakenews.repository.FakeNewsReportRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Service
public class FakeNewsReportService {

    @Autowired
    private FakeNewsReportRepository reportRepository;

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
        reportRepository.deleteById(id);
    }
}

