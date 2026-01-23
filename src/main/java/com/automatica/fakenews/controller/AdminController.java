package com.automatica.fakenews.controller;

import com.automatica.fakenews.model.FakeNewsReport;
import com.automatica.fakenews.model.ReportStatus;
import com.automatica.fakenews.service.FakeNewsReportService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.util.List;

@Controller
@RequestMapping("/admin")
public class AdminController {

    @Autowired
    private FakeNewsReportService reportService;

    @GetMapping("/dashboard")
    public String dashboard(Model model) {
        List<FakeNewsReport> pendingReports = reportService.getPendingReports();
        List<FakeNewsReport> inProgressReports = reportService.getInProgressReports();
        List<FakeNewsReport> approvedReports = reportService.getApprovedReports();
        List<FakeNewsReport> rejectedReports = reportService.getRejectedReports();

        model.addAttribute("pendingReports", pendingReports);
        model.addAttribute("inProgressReports", inProgressReports);
        model.addAttribute("approvedReports", approvedReports);
        model.addAttribute("rejectedReports", rejectedReports);

        return "admin/dashboard";
    }

    @PostMapping("/report/{id}/status/{status}")
    public String updateReportStatus(@PathVariable Long id,
                                     @PathVariable String status,
                                     Authentication authentication,
                                     RedirectAttributes redirectAttributes) {
        String username = authentication.getName();
        ReportStatus reportStatus = ReportStatus.valueOf(status.toUpperCase());
        reportService.setReportStatus(id, reportStatus, username);
        redirectAttributes.addFlashAttribute("successMessage", "Report status updated successfully!");
        return "redirect:/admin/dashboard";
    }

    @PostMapping("/report/{id}/in-progress")
    public String markAsInProgress(@PathVariable Long id,
                                   RedirectAttributes redirectAttributes) {
        reportService.setInProgressReport(id);
        redirectAttributes.addFlashAttribute("successMessage", "Report marked as in progress!");
        return "redirect:/admin/dashboard";
    }

    @PostMapping("/delete/{id}")
    public String deleteReport(@PathVariable Long id,
                               RedirectAttributes redirectAttributes) {
        reportService.deleteReport(id);
        redirectAttributes.addFlashAttribute("successMessage", "Report deleted successfully!");
        return "redirect:/admin/dashboard";
    }
}

