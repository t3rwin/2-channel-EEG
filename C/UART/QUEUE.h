/*
 * QUEUE.h
 *
 * Created on: Apr 22, 2024
 * Author: sebtu
 */

#ifndef INC_QUEUE_H_
#define INC_QUEUE_H_

#include <stdbool.h> // Include standard boolean library

#define MAX_SIZE 10  // Define the maximum size of the queue

// Queue structure definition
typedef struct {
    int items[MAX_SIZE];
    int front;
    int rear;
} Queue;

// Function prototypes
void initQueue(Queue *q); // Use pointer to modify the queue
bool isEmpty(Queue *q); // Use pointer for efficiency
bool isFull(Queue *q); // Use pointer for efficiency
void enqueue(Queue *q, int element); // Use pointer to modify the queue
int dequeue(Queue *q); // Use pointer to modify the queue
int peek(Queue *q); //look at first element in queue

#endif /* INC_QUEUE_H_ */
