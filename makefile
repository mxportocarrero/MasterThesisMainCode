APP_NAME = app

SRC_DIR = src/

FLAGS = `pkg-config opencv4 --cflags --libs` -mavx2 -O3

others = linear_algebra_functions.o utilities.o mean_shift.o kabsch.o
forest_objects = feature.o node.o tree.o forest.o
objects = direct_odometry.o dataset.o
systems = main_system_a.o main_system_b.o main_system_c.o main_system_d.o
app = main.o


%.o: $(SRC_DIR)%.cpp
	g++ $(FLAGS) -c -o $@ $<

%.o: %.cpp
	g++ $(FLAGS) -c -o $@ $<


all: $(systems) $(objects) $(forest_objects) $(others) $(app)
	g++ $(app) $(systems) $(objects) $(forest_objects) $(others) -o $(APP_NAME) $(FLAGS)

clean:
	rm -f *.o $(APP_NAME)