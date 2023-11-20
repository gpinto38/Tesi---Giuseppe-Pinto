package org.apache.commons.math4.neuralnet;
import org.apache.commons.math4.neuralnet.DistanceMeasure;
import org.apache.commons.math4.neuralnet.MapRanking;
import org.apache.commons.math4.neuralnet.Neuron;
import org.apache.commons.math4.neuralnet.internal.NeuralNetException;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;
import static org.junit.Assert.*;

import org.junit.Test;
import static org.junit.Assert.*;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class MapRankingBardTest {

    @Test
    public void testInitialization() {
        Set<Neuron> neurons = new HashSet<>();
        neurons.add(new Neuron(1, new double[]{1.0, 2.0, 3.0}));
        neurons.add(new Neuron(2, new double[]{4.0, 5.0, 6.0}));

        MapRanking mapRanking = new MapRanking(neurons, new EuclideanDistance());
        assertNotNull(mapRanking);
    }

    @Test
    public void testRank() {
        Set<Neuron> neurons = new HashSet<>();
        neurons.add(new Neuron(1, new double[]{1.0, 2.0, 3.0}));
        neurons.add(new Neuron(2, new double[]{4.0, 5.0, 6.0}));

        MapRanking mapRanking = new MapRanking(neurons, new EuclideanDistance());

        List<Neuron> rankedList = mapRanking.rank(new double[]{2.0, 3.0, 4.0});
        assertEquals(2, rankedList.size());
        assertEquals(1, rankedList.get(0).getIdentifier());
        assertEquals(2, rankedList.get(1).getIdentifier());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testRankInvalidInput() {
        Set<Neuron> neurons = new HashSet<>();
        neurons.add(new Neuron(1, new double[]{1.0, 2.0, 3.0}));
        neurons.add(new Neuron(2, new double[]{4.0, 5.0, 6.0}));

        MapRanking mapRanking = new MapRanking(neurons, new EuclideanDistance());
        mapRanking.rank(new double[]{1.0, 2.0});
    }

    @Test
    public void testRankWithMaxSize() {
        Set<Neuron> neurons = new HashSet<>();
        neurons.add(new Neuron(1, new double[]{1.0, 2.0, 3.0}));
        neurons.add(new Neuron(2, new double[]{4.0, 5.0, 6.0}));

        MapRanking mapRanking = new MapRanking(neurons, new EuclideanDistance());

        List<Neuron> rankedList = mapRanking.rank(new double[]{2.0, 3.0, 4.0}, 1);
        assertEquals(1, rankedList.size());
    }

    @Test(expected = NeuralNetException.class)
    public void testRankInvalidMaxSize() {
        Set<Neuron> neurons = new HashSet<>();
        neurons.add(new Neuron(1, new double[]{1.0, 2.0, 3.0}));
        neurons.add(new Neuron(2, new double[]{4.0, 5.0, 6.0}));

        MapRanking mapRanking = new MapRanking(neurons, new EuclideanDistance());
        mapRanking.rank(new double[]{1.0, 2.0, 3.0}, 0);
    }


}


