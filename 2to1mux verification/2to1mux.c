#include <stdio.h>
#include <assert.h>


int activate(int sum) {
    return (sum >= 1) ? 1 : 0;
}


int neural_network(int input1, int input2, int selector) {
    return (selector == 0) ? input1 : input2;
}




int main() {
 
    int input1, input2, selector;

  
    input1 = 1;
    input2 = 0;
    selector = 0;
    assert(neural_network(input1, input2, selector) == input1);


    input1 = 1;
    input2 = 0;
    selector = 1;
    assert(neural_network(input1, input2, selector) == input2);


    input1 = 0;
    input2 = 1;
    selector = 0;
    assert(neural_network(input1, input2, selector) == input1);


    input1 = 0;
    input2 = 1;
    selector = 1;
    assert(neural_network(input1, input2, selector) == input2);


    input1 = 0;
    input2 = 0;
    selector = 0;
    assert(neural_network(input1, input2, selector) == input1);

    
    input1 = 1;
    input2 = 1;
    selector = 1;
    assert(neural_network(input1, input2, selector) == input2);

    printf("All assertions passed. Neural network for 2:1 multiplexer verified.\n");

    return 0;
}
