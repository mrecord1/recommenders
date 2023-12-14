# Terraform configuration for deploying a Django app to Google Cloud Run
# with a Postgres database and a Cloud Storage bucket for static files.
# This configuration also creates a superuser for the Django app and
# stores the superuser password in Google Secret Manager.

# Activate Google Cloud
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.4.0"
    }
  }
}

# Set up data and variables
data "google_project" "project" {
    project_id  = var.project
}

variable "project" {
  description = "The ID of the project in which resources will be provisioned."
  type        = string
}

variable "region" {
  description = "The region in which resources will be provisioned."
  type        = string
  default     = "us-east1"
}

variable "service" {
  description = "The name of the service."
  type        = string
  default     = "django-mtg-service"
}

variable "repo" {
  description = "The name of the repository."
  type        = string
  default     = "django-mtg-repo"
}

provider "google" {
  project = var.project
  region  = var.region
}

# Activate service APIs
resource "google_project_service" "run" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "sql-component" {
  service            = "sql-component.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "sqladmin" {
  service            = "sqladmin.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "compute" {
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudbuild" {
  service            = "cloudbuild.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "secretmanager" {
  service            = "secretmanager.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifactregistry" {
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

# Create the service account
resource "google_service_account" "django" {
  account_id = "django"
}

# Create the database password, instance, db, and user
resource "random_password" "database_password" {
  length  = 32
  special = false
}

resource "google_sql_database_instance" "instance" {
    name             = "${var.project}-postgres"
    database_version = "POSTGRES_15"
    region           = var.region
    settings {
        tier = "db-f1-micro"
    }
    deletion_protection = true
}

resource "google_sql_database" "database" {
  name     = "${var.project}-db"
  instance = google_sql_database_instance.instance.name
}

resource "google_sql_user" "django" {
  name     = "${var.project}-user"
  instance = google_sql_database_instance.instance.name
  password = random_password.database_password.result
}

# Create the bucket for static files
resource "google_storage_bucket" "media" {
  name     = "${var.project}-media"
  location = "US"
}

# Manage secrets for django settings
resource "random_password" "django_secret_key" {
  special = false
  length  = 50
}

resource "google_secret_manager_secret" "django_settings" {
  secret_id = "django_settings"

  replication {
    auto {}
  }
  depends_on = [google_project_service.secretmanager]

}

resource "google_secret_manager_secret_version" "django_settings" {
  secret = google_secret_manager_secret.django_settings.id

  secret_data = templatefile("etc/env.tpl", {
    bucket     = google_storage_bucket.media.name
    secret_key = random_password.django_secret_key.result
    user       = google_sql_user.django
    instance   = google_sql_database_instance.instance
    database   = google_sql_database.database
  })
}

# Manage IAM for django settings
resource "google_secret_manager_secret_iam_binding" "django_settings" {
  secret_id = google_secret_manager_secret.django_settings.id
  role      = "roles/secretmanager.secretAccessor"
  members   = [local.cloudbuild_serviceaccount, local.django_serviceaccount]
}

locals {
  cloudbuild_serviceaccount = "serviceAccount:${data.google_project.project.number}@cloudbuild.gserviceaccount.com"
  django_serviceaccount     = "serviceAccount:${google_service_account.django.email}"
}

# Manage superuser password
resource "random_password" "superuser_password" {
  length  = 32
  special = false
}

resource "google_secret_manager_secret" "superuser_password" {
  secret_id = "superuser_password"
  replication {
    auto {}
  }
  depends_on = [google_project_service.secretmanager]
}

resource "google_secret_manager_secret_version" "superuser_password" {
  secret      = google_secret_manager_secret.superuser_password.id
  secret_data = random_password.superuser_password.result
}

resource "google_secret_manager_secret_iam_binding" "superuser_password" {
  secret_id = google_secret_manager_secret.superuser_password.id
  role      = "roles/secretmanager.secretAccessor"
  members   = [local.cloudbuild_serviceaccount]
}

# Create the cloud artifact registry repository
resource "google_artifact_registry_repository" "repository" {
location      = var.region
repository_id = var.repo
description   = "Repository for ${var.service}" 
format        = "DOCKER"
}

# Create the cloud run service
resource "google_cloud_run_v2_service" "service" {
  name                       = var.service
  location                   = var.region
  ingress                    = "INGRESS_TRAFFIC_INTERNAL_ONLY"

  template {
    scaling {
      max_instance_count = 1
    }

    service_account = google_service_account.django.email

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project}/${var.repo}/${var.service}"
      
      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }
    }
 
    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = ["${var.project}:${var.region}:${var.service}-postgres"]
      }
    }
  }
  client     = "terraform"
  depends_on = [google_project_service.secretmanager, google_project_service.run, google_project_service.sqladmin]
  
  traffic {
    type            = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent         = 100
  }
}

data "google_iam_policy" "noauth" {
  binding {
    role = "roles/run.invoker"
    members = [
      "allUsers",
    ]
  }
}

resource "google_cloud_run_service_iam_policy" "noauth" {
  location = google_cloud_run_v2_service.service.location
  project  = google_cloud_run_v2_service.service.project
  service  = google_cloud_run_v2_service.service.name

  policy_data = data.google_iam_policy.noauth.policy_data
}

# Create database access
resource "google_project_iam_binding" "service_permissions" {
  project = var.project
  for_each = toset([
    "run.admin", "cloudsql.client"
  ])

  role    = "roles/${each.key}"
  members = [local.cloudbuild_serviceaccount, local.django_serviceaccount]

}

resource "google_service_account_iam_binding" "cloudbuild_sa" {
  service_account_id = google_service_account.django.name
  role               = "roles/iam.serviceAccountUser"

  members = [local.cloudbuild_serviceaccount]
}

# View service at url
output "superuser_password" {
  value     = google_secret_manager_secret_version.superuser_password.secret_data
  sensitive = true
}

output "service_url" {
  value = google_cloud_run_v2_service.service.status[0].url
}
