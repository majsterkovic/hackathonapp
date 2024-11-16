# define base docker image
# Wybierz obraz bazowy z Mavenem
FROM maven:latest

# Ustaw katalog roboczy w kontenerze
WORKDIR /app

# Skopiuj pliki aplikacji do kontenera
COPY src /app/src
COPY pom.xml /app/pom.xml

# Wykonaj polecenie "mvn clean install" w kontenerze
RUN mvn clean install -Dmaven.test.skip=true

# Wykorzystaj obraz bazowy OpenJDK do uruchomienia aplikacji
FROM openjdk:17

# Skopiuj skompilowaną aplikację do kontenera
COPY --from=0 /app/target/hackathonapp-0.0.1-SNAPSHOT.jar hackathonapp-jar

# Ustaw etykietę i punkt wejścia
LABEL maintainer="pyrai"
ENTRYPOINT ["java", "-jar", "hackathonapp-jar"]