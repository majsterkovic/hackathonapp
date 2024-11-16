package ai.pyr.hackathonapp.entities;

import jakarta.persistence.*;
import lombok.Getter;

import java.io.Serializable;

@Entity
@Table(name = "app_user")
public class UserEntity implements Serializable {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id", nullable = false)
    private Long id;

    @Getter
    @Column(name = "name", nullable = false)
    private String name;

    public UserEntity(String name) {
        this.name = name;
    }

    public UserEntity() {

    }

}
