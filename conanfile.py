from conans import ConanFile, CMake, tools


class HkgConan(ConanFile):
  name = "hkg"
  version = "0.0.1"
  license = "MIT"
  author = "zhengqihang 18000632@smail.cczu.edu.cn"
  url = "<Package recipe repository url here, for issues about the package>"
  description = "Halide kernel generator"
  topics = ("DNN", "compiler")
  settings = "os", "compiler", "build_type", "arch"
  options = {"shared": [True, False],
             "fPIC": [True, False]}

  default_options = {"shared": False,
                     "fPIC": True}
                     
  generators = ["cmake", "cmake_find_package", "cmake_paths"]

  exports_sources = ['benchmark/*',
                     'include/*',
                     'tests/*',
                     'CMakeLists.txt',
                     'kernels_generator.cpp',
                     'README.md']

  def requirements(self):
    self.requires("Halide/12.0.0")

  def config_options(self):
    if self.settings.os == "Windows":
      del self.options.fPIC

  def configure(self):
    pass

  def source(self):
    pass

  def cmake_configure(self):
    cmake = CMake(self)
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
