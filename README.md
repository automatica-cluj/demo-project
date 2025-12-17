# Fake News Reporter

[![CI](https://github.com/automatica-cluj/demo-project/actions/workflows/ci.yml/badge.svg)](https://github.com/automatica-cluj/demo-project/actions/workflows/ci.yml)
[![Build and Push](https://github.com/automatica-cluj/demo-project/actions/workflows/build-push.yml/badge.svg)](https://github.com/automatica-cluj/demo-project/actions/workflows/build-push.yml)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/automatica-cluj/demo-project)](https://github.com/automatica-cluj/demo-project/releases)
[![Docker Image](https://img.shields.io/badge/docker-ghcr.io-blue)](https://github.com/automatica-cluj/demo-project/pkgs/container/demo-project)

A Spring Boot application for reporting and managing fake news sources. Users can report suspicious news sources, and administrators can verify and approve reports for public visibility!!!!!

**This project is educational, for demonstration purposes only!**

![infographics](infographics.png)

## Features

- **Public Features:**
  - View verified fake news reports
  - Report new fake news sources
  - Browse reports by category (Politics, Health, Science, Technology, Entertainment, Finance)

- **Admin Features:**
  - Secure login system
  - Review pending reports
  - Approve or reject reports
  - Delete inappropriate submissions

## Technology Stack

- **Backend:** Spring Boot 3.2.0
- **Frontend:** Thymeleaf with custom CSS
- **Security:** Spring Security with BCrypt password encoding
- **Database:** 
  - H2 (in-memory) for local development
  - PostgreSQL for production deployment
- **Build Tool:** Maven
- **Containerization:** Docker & Docker Compose

## Prerequisites

- Java 17 or higher
- Maven 3.6+
- Docker & Docker Compose (for production deployment)

## Running Locally (H2 Database)

1. Clone the repository:
```bash
git clone https://github.com/automatica-cluj/demo-project.git
cd demo-project
```

2. Build and run the application:
```bash
mvn spring-boot:run
```

3. Access the application:
   - Main application: http://localhost:8080
   - H2 Console: http://localhost:8080/h2-console
     - JDBC URL: `jdbc:h2:mem:fakenews`
     - Username: `sa`
     - Password: (leave empty)

4. Login credentials:
   - Username: `admin`
   - Password: `admin123`

## Running with Docker Compose (PostgreSQL)

1. Build and start the containers:
```bash
docker-compose up --build
```

2. Access the application:
   - Main application: http://localhost:8080

3. Stop the application:
```bash
docker-compose down
```

4. To remove all data (volumes):
```bash
docker-compose down -v
```

## Application Structure

```
src/main/java/com/automatica/fakenews/
├── config/              # Configuration classes (Security, Data initialization)
├── controller/          # Web controllers (Home, Admin)
├── dto/                 # Data Transfer Objects
├── model/               # JPA Entity classes
├── repository/          # Spring Data repositories
├── service/             # Business logic services
└── FakeNewsReporterApplication.java

src/main/resources/
├── static/css/          # CSS stylesheets
├── templates/           # Thymeleaf templates
│   ├── admin/          # Admin-specific templates
│   └── ...             # Public templates
├── application.yml      # Main configuration
├── application-local.yml    # H2 configuration
└── application-prod.yml     # PostgreSQL configuration
```

## API Endpoints

### Public Endpoints
- `GET /` - Home page with verified reports
- `GET /reports` - View all verified reports
- `GET /report` - Report submission form
- `POST /report` - Submit a new report
- `GET /login` - Admin login page

### Admin Endpoints (Authentication Required)
- `GET /admin/dashboard` - Admin dashboard
- `POST /admin/approve/{id}` - Approve a report
- `POST /admin/delete/{id}` - Delete a report

## Database Schema

### users
- `id` (BIGINT, Primary Key)
- `username` (VARCHAR, Unique)
- `password` (VARCHAR, BCrypt encoded)
- `role` (VARCHAR)
- `enabled` (BOOLEAN)

### fake_news_reports
- `id` (BIGINT, Primary Key)
- `news_source` (VARCHAR)
- `url` (VARCHAR)
- `category` (VARCHAR)
- `description` (TEXT)
- `reported_at` (TIMESTAMP)
- `approved` (BOOLEAN)
- `approved_at` (TIMESTAMP)
- `approved_by` (VARCHAR)

## Configuration Profiles

- **local** (default): Uses H2 in-memory database
- **prod**: Uses PostgreSQL database

Switch profiles using:
```bash
# Command line
mvn spring-boot:run -Dspring-boot.run.profiles=prod

# Environment variable
export SPRING_PROFILE=prod
```

## Building for Production

Build the JAR file:
```bash
mvn clean package
```

Run the JAR:
```bash
java -jar target/fake-news-reporter-1.0.0.jar
```

## Environment Variables (Production)

- `SPRING_PROFILE` - Active profile (prod)
- `DB_HOST` - PostgreSQL host (default: localhost)
- `DB_PORT` - PostgreSQL port (default: 5432)
- `DB_NAME` - Database name (default: fakenews)
- `DB_USER` - Database username (default: postgres)
- `DB_PASSWORD` - Database password (default: postgres)

## Security Notes

- The default admin password should be changed in production
- Passwords are stored using BCrypt hashing
- CSRF protection is enabled for all forms
- Spring Security protects admin endpoints

## License

This project is for demonstration purposes.

## Contributing

Feel free to submit issues and pull requests.
