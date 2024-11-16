package ai.pyr.hackathonapp.controllers;

import jakarta.servlet.http.HttpServletRequest;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("first-web")
public class firstController {

    @GetMapping("/my_name")
    public String getName(HttpServletRequest request) {
        String token = request.getHeader("Authorization");
        System.out.println("Twoj token to : " + token );
        return "Hello, World 2!";
        
    }
}
