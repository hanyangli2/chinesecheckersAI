package org.deeplearning4j.rl4j.space;

import org.nd4j.linalg.api.rng.Random;

public abstract class VariableDiscreteSpace extends DiscreteSpace
{
    public VariableDiscreteSpace()
    {
        super(0);
    }

    public VariableDiscreteSpace(Random rnd)
    {
        super(0, rnd);
    }

    abstract public int getSize();

    public Integer randomAction() {
        return rnd.nextInt(getSize());
    }

    public Object encode(Integer a) {
        return a;
    }

    public Integer noOp() {
        return 0;
    }

}
