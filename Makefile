CXXFLAGS = -O3 -Wall -Wextra
LIBS = -lSDL2 -lSDL2main

SRC = main.cpp model.cpp texture.cpp tgaimage.cpp
OBJS  := $(patsubst %.cpp, %.o, $(SRC))

all: build/cpu-rasterizer

build/cpu-rasterizer: $(OBJS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) -o build/cpu-rasterizer $(OBJS) $(LIBS) 

clean:
	rm -rf build



