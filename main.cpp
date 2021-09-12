#include "simMain.h"

int main(int argc, char **argv) {
#ifdef SIMVIZ
    simMainViz(argc, argv);
#else
    simMainNoViz(argc, argv);
#endif // SIMVIZ
}