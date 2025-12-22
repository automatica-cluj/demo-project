package com.automatica.fakenews.controller;

import com.automatica.fakenews.model.FakeNewsReport;
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
        List<FakeNewsReport> approvedReports = reportService.getApprovedReports();
        List<FakeNewsReport> rejectedReports = reportService.getRejectedReports();
        
        model.addAttribute("pendingReports", pendingReports);
        model.addAttribute("approvedReports", approvedReports);
        model.addAttribute("rejectedReports", rejectedReports);
        
        return "admin/dashboard";
    }

    @PostMapping("/approve/{id}")
    public String approveReport(@PathVariable Long id, 
                                Authentication authentication,
                                RedirectAttributes redirectAttributes) {
        String username = authentication.getName();
        reportService.approveReport(id, username);
        redirectAttributes.addFlashAttribute("successMessage", "Report approved successfully!");
        return "redirect:/admin/dashboard";
    }

    @PostMapping("/reject/{id}")
    public String rejectReport(@PathVariable Long id, 
                               RedirectAttributes redirectAttributes) {
        reportService.rejectReport(id);
        redirectAttributes.addFlashAttribute("successMessage", "Report rejected successfully!");
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
