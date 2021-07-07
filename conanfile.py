from conans import ConanFile, CMake, tools
import os
import shutil


class HkgConan(ConanFile):
  name = "hkg"
  version = "0.0.1"
  license = "MIT"
  author = "zhengqihang 18000632@smail.cczu.edu.cn"
  url = "<Package recipe repository url here, for issues about the package>"
  description = "Halide kernel generator"
  topics = ("DNN", "compiler")
  settings = "arch"
  options = {"tests": [True, False],
             "benchmark": [True, False]}

  default_options = {"tests": False,
                     "benchmark": False}

  generators = ["cmake", "cmake_find_package", "cmake_paths"]

  exports_sources = ['src/*',
                     'include/*',
                     'cmake/*',
                     'CMakeLists.txt']

  def build_requirements(self):
    self.build_requires("Halide/12.0.0")

  def config_options(self):
    if self.settings.arch != "x86_64":
      raise ValueError("not support other patfrom!")

  def configure(self):
    pass

  def source(self):
    pass

  def cmake_configure(self):
    cmake = CMake(self)
    cmake.definitions['ENABLE_BENCHMARK'] = self.options.benchmark
    cmake.definitions['ENABLE_TEST'] = self.options.tests
    cmake.configure()
    return cmake

  def build(self):
    cmake = self.cmake_configure()
    cmake.build()

  def package(self):
    cmake = self.cmake_configure()
    cmake.install()

  def package_info(self):
    pass
