cmake_minimum_required(VERSION 2.8.3)

find_package(catkin REQUIRED)
catkin_package()
catkin_python_setup()

# The trailing slash is necessary to prevent CMake from creating a "data"
# directory inside DESTINATION.
# See: http://www.cmake.org/cmake/help/v3.0/command/install.html
install(DIRECTORY "${PROJECT_SOURCE_DIR}/ordata/"
    DESTINATION "${OpenRAVE_INSTALL_DIR}/${OpenRAVE_DATA_DIR}"
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(tests/test_MoveHand.py)
endif(CATKIN_ENABLE_TESTING)
