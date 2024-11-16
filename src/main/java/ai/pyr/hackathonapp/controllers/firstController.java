package ai.pyr.hackathonapp.controllers;

import ai.pyr.hackathonapp.entities.UserEntity;
import ai.pyr.hackathonapp.repositories.UserRepository;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("first-web")
public class firstController {
    private final UserRepository userRepository;

    public firstController(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    @GetMapping("/my_name")
    public String getName(HttpServletRequest request) {
        String token = request.getHeader("Authorization");
        System.out.println("Twoj token to : " + token );
        List<UserEntity> users = userRepository.findAll();
        StringBuilder res = new StringBuilder();
        for (UserEntity user: users) {
            res.append(user.getName());
        }
        return res.toString();
    }

    @GetMapping("/add_user")
    public String addUser(HttpServletRequest request) {
        String token = request.getHeader("Authorization");
        userRepository.save(new UserEntity("abc"));
        return "dodano";
    }
}
