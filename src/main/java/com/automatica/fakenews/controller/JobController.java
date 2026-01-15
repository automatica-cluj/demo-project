package com.automatica.fakenews.controller;

import com.automatica.fakenews.model.Job;
import com.automatica.fakenews.service.JobService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.util.List;

@Controller
@RequestMapping("/admin/jobs")
public class JobController {

    @Autowired
    private JobService jobService;

    @GetMapping
    public String listJobs(Model model) {
        List<Job> pendingJobs = jobService.getPendingJobs();
        List<Job> approvedJobs = jobService.getApprovedJobs();
        List<Job> completedJobs = jobService.getCompletedJobs();
        
        model.addAttribute("pendingJobs", pendingJobs);
        model.addAttribute("approvedJobs", approvedJobs);
        model.addAttribute("completedJobs", completedJobs);
        
        return "admin/jobs";
    }

    @PostMapping("/approve/{id}")
    public String approveJob(@PathVariable Long id, 
                            Authentication authentication,
                            RedirectAttributes redirectAttributes) {
        String username = authentication.getName();
        jobService.approveJob(id, username);
        redirectAttributes.addFlashAttribute("successMessage", 
            "Job approved and scheduled for execution!");
        return "redirect:/admin/jobs";
    }

    @PostMapping("/reject/{id}")
    public String rejectJob(@PathVariable Long id, 
                           Authentication authentication,
                           RedirectAttributes redirectAttributes) {
        String username = authentication.getName();
        jobService.rejectJob(id, username);
        redirectAttributes.addFlashAttribute("successMessage", "Job rejected successfully!");
        return "redirect:/admin/jobs";
    }
}
