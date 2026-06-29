group "default" {
  targets = ["dataset", "web"]
}

target "dataset" {
  context = "."
  dockerfile = "docker/Dockerfile.dataset"
  tags = ["social-safety-amr/dataset-service:local"]
}

target "web" {
  context = "."
  dockerfile = "docker/Dockerfile.web"
  tags = ["social-safety-amr/web-gui:local"]
}

target "geometry" {
  context = "."
  dockerfile = "docker/Dockerfile.geometry"
  tags = ["social-safety-amr/geometry-service:local"]
}
