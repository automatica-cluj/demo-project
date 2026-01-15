package com.automatica.fakenews.service;

import com.automatica.fakenews.model.FakeNewsReport;
import com.automatica.fakenews.repository.FakeNewsReportRepository;
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
class FakeNewsReportServiceTest {

    @Mock
    private FakeNewsReportRepository reportRepository;

    @Mock
    private JobService jobService;

    @InjectMocks
    private FakeNewsReportService reportService;

    @Test
    void testSaveReport_Success() {
        // Given
        FakeNewsReport report = new FakeNewsReport();
        report.setNewsSource("Fake News Daily");
        report.setUrl("http://fakenews.com");
        report.setCategory("Politics");
        report.setDescription("This is a fake news source");

        FakeNewsReport savedReport = new FakeNewsReport();
        savedReport.setId(1L);
        savedReport.setNewsSource("Fake News Daily");
        savedReport.setUrl("http://fakenews.com");
        savedReport.setCategory("Politics");
        savedReport.setDescription("This is a fake news source");

        when(reportRepository.save(any(FakeNewsReport.class))).thenReturn(savedReport);

        // When
        FakeNewsReport result = reportService.saveReport(report);

        // Then
        assertNotNull(result);
        verify(reportRepository, times(1)).save(report);
        assertEquals("Fake News Daily", result.getNewsSource());
        
        // Verify job was created
        verify(jobService, times(1)).createJob(eq("REPORT_VERIFICATION"), anyString());
    }

    @Test
    void testApproveReport_SetsAllFields() {
        // Given
        FakeNewsReport report = new FakeNewsReport();
        report.setId(1L);
        report.setNewsSource("Fake News Daily");
        report.setUrl("http://fakenews.com");
        report.setCategory("Politics");
        report.setApproved(false);

        when(reportRepository.findById(1L)).thenReturn(Optional.of(report));
        when(reportRepository.save(any(FakeNewsReport.class))).thenReturn(report);

        // When
        reportService.approveReport(1L, "admin");

        // Then
        ArgumentCaptor<FakeNewsReport> captor = ArgumentCaptor.forClass(FakeNewsReport.class);
        verify(reportRepository).save(captor.capture());

        FakeNewsReport savedReport = captor.getValue();
        assertTrue(savedReport.isApproved(), "Report should be approved");
        assertEquals("admin", savedReport.getApprovedBy(), "Approved by should be set to 'admin'");
        assertNotNull(savedReport.getApprovedAt(), "Approved at timestamp should be set");
    }

    @Test
    void testApproveReport_ReportNotFound_DoesNothing() {
        // Given
        when(reportRepository.findById(999L)).thenReturn(Optional.empty());

        // When
        reportService.approveReport(999L, "admin");

        // Then
        verify(reportRepository, never()).save(any(FakeNewsReport.class));
    }

    @Test
    void testGetApprovedReports_DelegatesToRepository() {
        // Given
        FakeNewsReport report1 = new FakeNewsReport();
        report1.setId(1L);
        report1.setNewsSource("Source 1");
        report1.setApproved(true);

        FakeNewsReport report2 = new FakeNewsReport();
        report2.setId(2L);
        report2.setNewsSource("Source 2");
        report2.setApproved(true);

        List<FakeNewsReport> approvedReports = Arrays.asList(report1, report2);
        when(reportRepository.findByApprovedTrueOrderByApprovedAtDesc()).thenReturn(approvedReports);

        // When
        List<FakeNewsReport> result = reportService.getApprovedReports();

        // Then
        assertNotNull(result);
        assertEquals(2, result.size());
        assertTrue(result.get(0).isApproved());
        assertTrue(result.get(1).isApproved());
        verify(reportRepository, times(1)).findByApprovedTrueOrderByApprovedAtDesc();
    }

    @Test
    void testGetPendingReports_DelegatesToRepository() {
        // Given
        FakeNewsReport report1 = new FakeNewsReport();
        report1.setId(1L);
        report1.setNewsSource("Pending Source 1");
        report1.setApproved(false);

        FakeNewsReport report2 = new FakeNewsReport();
        report2.setId(2L);
        report2.setNewsSource("Pending Source 2");
        report2.setApproved(false);

        List<FakeNewsReport> pendingReports = Arrays.asList(report1, report2);
        when(reportRepository.findByApprovedFalseOrderByReportedAtDesc()).thenReturn(pendingReports);

        // When
        List<FakeNewsReport> result = reportService.getPendingReports();

        // Then
        assertNotNull(result);
        assertEquals(2, result.size());
        assertFalse(result.get(0).isApproved());
        assertFalse(result.get(1).isApproved());
        verify(reportRepository, times(1)).findByApprovedFalseOrderByReportedAtDesc();
    }
}
