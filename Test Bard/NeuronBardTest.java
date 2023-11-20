package org.apache.commons.math4.neuralnet;
import org.apache.commons.math4.neuralnet.Neuron;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.*;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@Execution(ExecutionMode.CONCURRENT)
class NeuronBardTest {

    private Neuron neuron1;

    @BeforeEach
    void setUp() {
        neuron1 = new Neuron(1, new double[]{1.0, 2.0, 3.0});
    }

    @Test
    void testCopy() {
        Neuron neuron2 = neuron1.copy();

        assertEquals(neuron1.getIdentifier(), neuron2.getIdentifier());
        assertEquals(neuron1.getSize(), neuron2.getSize());
        assertArrayEquals(neuron1.getFeatures(), neuron2.getFeatures());
    }

    @Test
    void testCopyConcurrency() throws InterruptedException {
        int numThreads = 10;
        Thread[] threads = new Thread[numThreads];

        for (int i = 0; i < numThreads; i++) {
            threads[i] = new Thread(() -> {
                Neuron neuron2 = neuron1.copy();

                assertEquals(neuron1.getIdentifier(), neuron2.getIdentifier());
                assertEquals(neuron1.getSize(), neuron2.getSize());
                assertArrayEquals(neuron1.getFeatures(), neuron2.getFeatures());
            });
        }

        for (Thread thread : threads) {
            thread.start();
        }

        for (Thread thread : threads) {
            thread.join();
        }
    }

    @Test
    void testGetIdentifier() {
        assertEquals(1, neuron1.getIdentifier());
    }

    @Test
    void testGetSize() {
        assertEquals(3, neuron1.getSize());
    }

    @Test
    void testGetFeatures() {
        double[] features = neuron1.getFeatures();

        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, features);
    }

    @Test
    void testCompareAndSetFeatures() {
        double[] expectedFeatures = new double[]{4.0, 5.0, 6.0};
        double[] currentFeatures = neuron1.getFeatures();

        assertTrue(neuron1.compareAndSetFeatures(currentFeatures, expectedFeatures));
        assertArrayEquals(expectedFeatures, neuron1.getFeatures());

        double[] invalidFeatures = new double[]{7.0, 8.0, 9.0};

        assertFalse(neuron1.compareAndSetFeatures(currentFeatures, invalidFeatures));
        assertArrayEquals(expectedFeatures, neuron1.getFeatures());
    }

    @Test
    void testCompareAndSetFeaturesConcurrency() throws InterruptedException {
        int numThreads = 10;
        Thread[] threads = new Thread[numThreads];
        ReentrantLock lock = new ReentrantLock();

        for (int i = 0; i < numThreads; i++) {
            threads[i] = new Thread(() -> {
                double[] expectedFeatures = new double[]{10.0, 11.0, 12.0};
                double[] currentFeatures;

                lock.lock(); // Acquisisci il lock
                try {
                    currentFeatures = neuron1.getFeatures();

                    boolean success = neuron1.compareAndSetFeatures(currentFeatures, expectedFeatures);

                    if (success) {
                        assertArrayEquals(expectedFeatures, neuron1.getFeatures());
                    } else {
                        assertArrayEquals(currentFeatures, neuron1.getFeatures());
                    }
                } finally {
                    lock.unlock(); // Rilascia il lock
                }
            });
        }

        for (Thread thread : threads) {
            thread.start();
        }

        for (Thread thread : threads) {
            thread.join();
        }
    }
}
