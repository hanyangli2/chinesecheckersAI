package org.deeplearning4j.rl4j.examples.advanced.src.main.java;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDenseVarActions;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.configuration.DQNDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.network.dqn.DQNVarActionsFactoryStdDense;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.deeplearning4j.rl4j.space.VariableDiscreteSpace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.deeplearning4j.rl4j.policy.DQNPolicy;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.*;
import java.util.ArrayList;

import static org.deeplearning4j.rl4j.examples.advanced.src.main.java.GameBoard.EvaluatePath;

public class ChineseCheckersTraining {
    static final int n = 7;
    static final int K = 3000; // #steps to play the random opponent
    static final int m = 3; //longest length of starting pieces... 1 = 1 piece, 2 = 3 piece, 3 = 6 piece, 4 = 10 piece... etc.
    static Random rnd = new Random(222);

    static class SineObservationSpace implements ObservationSpace<State> {
        public INDArray getLow() {
            return null; // deprecated
        }

        public INDArray getHigh() {
            return null; // deprecated
        }

        public String getName() {
            return "Variable Discrete";
        }

        // shape of the state-action representation; in this example it is 2 for a pair of numbers (action_id, sin(time))
        public int[] getShape() {
            return new int[]{n * n + 3};
        }
    }

    static class Env implements MDP<State, Integer, DiscreteSpace> {
        int time = 0;
        int totalSteps = 0;

        private final GameBoard gameBoard;

        public Env(GameBoard gameBoard) {
            this.gameBoard = gameBoard;
        }

        private final VariableDiscreteSpace actionSpace = new VariableDiscreteSpace() {
            @Override
            public int getSize() {
                return gameBoard.getAvailableActionsCount();
            }
        };

        public ObservationSpace<State> getObservationSpace() {
            return new SineObservationSpace(); //board obseravation space
        }

        public DiscreteSpace getActionSpace() {
            return actionSpace;
        }

        public StepReply<State> step(Integer action) {
            double reward = 0.0;

            //q learning move
            ArrayList<String> ChosenMove = gameBoard.RLAgentStep(action);
            double RLReward = EvaluatePath(ChosenMove);

            if (gameBoard.CheckForWin() == 1) {
                //reward = 10;
                reward = 10.0 * (1 + Math.exp(-time / 5.0));
                System.out.println("PLayer 1 won in " + time + " steps; avgDistance1=" + String.format("%.1f", gameBoard.avgDistance(gameBoard.getPlayerPositions(), 1))
                    + ";  avgDistance1=" + String.format("%.1f", gameBoard.avgDistance(gameBoard.getPlayerPositions(), 2)));
            } else {
                time += 1;

                if (totalSteps < K) {
                    //random move
                    ArrayList<String> randomMove = gameBoard.RandomAgentStep(2);
                    double opponentReward = EvaluatePath(randomMove);
                    reward = (RLReward + opponentReward) * 0.5;
                } else {
                    //greedy agent move
                    ArrayList<String> greedyMove = gameBoard.GreedyAgentStep(2);
                    double opponentReward = EvaluatePath(greedyMove);
                    reward = (RLReward + opponentReward) * 0.5;
                }

                if (gameBoard.CheckForWin() == 2) {
                    //reward = -10;
                    reward = -10.0 * (1 + Math.exp(-time / 5.0));
                    System.out.println("PLayer 2 won in " + time + " steps; avgDistance1=" + String.format("%.1f", gameBoard.avgDistance(gameBoard.getPlayerPositions(), 1))
                        + ";  avgDistance1=" + String.format("%.1f", gameBoard.avgDistance(gameBoard.getPlayerPositions(), 2)));
                } else {
                    // update state
                    time += 1;
                }
            }

            totalSteps++;

            State newState = gameBoard.state();
            return new StepReply<>(newState, reward, isDone(), null);
        }

        public boolean isDone() {
            if (gameBoard.CheckForWin() == 1 || gameBoard.CheckForWin() == 2 || time > 600) {
                return true;
            } else {
                return false;
            }
        }


        public State reset() {
            gameBoard.reset();
            time = 0;
            return gameBoard.state();
        }

        public void close() {
        }

        public Env newInstance() {
            return create();
        }

        static Env create() {
            GameBoard gameBoard = new GameBoard(n, m);
            return new Env(gameBoard);
        }

    }

    public static void main(String[] args) {
        QLearningConfiguration qlc = QLearningConfiguration.builder()
            .seed(123L)
            .maxEpochStep(1000) //epoch cap
            .maxStep(250000) //step cap
            .expRepMaxSize(10000) //how much memory
            .batchSize(32)
            .targetDqnUpdateFreq(300) //how often update network
            .updateStart(0)
            //.rewardFactor(0.05)
            .gamma(0.99)
            .errorClamp(10.0)
            .minEpsilon(0.1) //exploration-exploitation
            .epsilonNbStep(2000)
            .doubleDQN(true)
            .build();


        DQNDenseNetworkConfiguration net = DQNDenseNetworkConfiguration.builder()
            .numHiddenNodes(50) //nodes in each layer of nueral net.
            .numLayers(3)
            .l2(0.01)
            .updater(new Adam(1e-2))
            .build();


        MDP<State, Integer, DiscreteSpace> mdp = Env.create();
        QLearningDiscreteDenseVarActions dql = new QLearningDiscreteDenseVarActions(mdp, new DQNVarActionsFactoryStdDense(net), qlc);
        dql.train();
        mdp.close();

        try {
            dql.getPolicy().save("C:\\Users\\huiwe\\Desktop\\20202021\\capstone\\trained agents\\7x7_6P_50N_3L_100S.zip");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
