#include "QUEUE.h"

// Initialize the queue
void initQueue(Queue *q) {
    q->front = -1;
    q->rear = -1;
}

// Check if the queue is empty
bool isEmpty(Queue *q) {
    return q->rear == -1;
}

// Check if the queue is full
bool isFull(Queue *q) {
    return q->rear == MAX_SIZE - 1;
}

// Add an element to the end of the queue
void enqueue(Queue *q, int element) {
    if (isFull(q)) {
    } else {
        if (isEmpty(q)) {
            q->front = 0;
        }
        q->rear++;
        q->items[q->rear] = element;
    }
}

// Remove an element from the front of the queue
int dequeue(Queue *q) {
    if (isEmpty(q)) {
        return -1;
    } else {
        int item = q->items[q->front];
        q->front++;
        if (q->front > q->rear) {
            // Reset queue after last dequeue
            initQueue(q);
        }
        return item;
    }
}

int peek(Queue *q) {
    if (isEmpty(q)) {
        return -1;  // Return -1 or any invalid value indicating the queue is empty
    }
    return q->items[q->front];
}
