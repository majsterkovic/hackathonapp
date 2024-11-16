package ai.pyr.hackathonapp.repositories;

import ai.pyr.hackathonapp.entities.UserEntity;
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserRepository extends JpaRepository<UserEntity, Long> {
}
