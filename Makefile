OBJECTS=test/test.cu src/cuda-project.cu src/cuda-project-cpu.cu

all: $(OBJECTS)
	nvcc $(OBJECTS) -o cuda-project

%.o: %.cpp
	nvcc -x cu -I. -dc $< -o $@

clean:
	rm -rf *.o cuda-project
