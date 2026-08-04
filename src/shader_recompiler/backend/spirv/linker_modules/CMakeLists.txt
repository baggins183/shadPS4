#TODO do this a better way without python or w/e


set(SHADER_FILES
    ordered_count_spv_lib.comp
)

foreach(FILENAME IN ITEMS ${SHADER_FILES})
    string(REPLACE "." "_" SHADER_NAME ${FILENAME})
    set(SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME})
    set(SPV_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv)
    set(HEADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.h)

    add_custom_command(
        OUTPUT
            ${SPV_FILE}
        COMMAND
            glslang-standalone -V ${SOURCE_FILE} -o ${SPV_FILE} --target-env vulkan1.4 --no-link
        DEPENDS
            ${SOURCE_FILE}
    )

    add_custom_command(
        OUTPUT
            ${HEADER_FILE}
        COMMAND
            Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/spv_to_h.py ${SPV_FILE} ${HEADER_FILE} ${SHADER_NAME}
        DEPENDS
            ${SPV_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/spv_to_h.py
    )

    set(LINKER_SPV_MODULE_HEADERS ${LINKER_SPV_MODULE_HEADERS} ${HEADER_FILE})
endforeach()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

if (NOT TARGET glslang-standalone)
    message(FATAL_ERROR "glslang::glslang-standalone does not exist!")
endif()

set(LINKER_SPV_MODULES_INCLUDE ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)

add_custom_target(linker_spv_modules
    DEPENDS
        ${LINKER_SPV_MODULE_HEADERS}
        glslang-standalone
    SOURCES
        ${SHADER_FILES}
)