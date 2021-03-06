from conans import ConanFile, CMake, tools
import os
import shutil


class HkgConan(ConanFile):
  name = "hkg"
  version = "0.0.3"
  license = "MIT"
  author = "zhengqihang 18000632@smail.cczu.edu.cn"
  url = "<Package recipe repository url here, for issues about the package>"
  description = "Halide kernel generator"
  topics = ("DNN", "compiler")
  settings = None
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
    self.options['Halide'].shared = True

  def config_options(self):
    pass

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
    self.cpp_info.build_modules = ['lib/cmake/hkgTargets.cmake', 'lib/cmake/hkgHelper.cmake']
