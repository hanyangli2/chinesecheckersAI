package org.deeplearning4j.rl4j.examples.advanced.src.main.java;
import java.io.IOException;
import java.util.ArrayList;

import org.deeplearning4j.rl4j.network.dqn.DQNVarActions;
import org.deeplearning4j.rl4j.policy.DQNPolicy;

public class RLVSRandom {
    static final int n = 7;
    static final int m = 3; //longest length of starting pieces... 1 = 1 piece, 2 = 3 piece, 3 = 6 piece, 4 = 10` piece... etc.
    static int step = 0;

    public static void main(String[] args){

        GameBoard gameboard = new GameBoard(n, m);

        DQNPolicy newPolicy = null;
        try {
            newPolicy = new DQNPolicy(DQNVarActions.load("C:\\Users\\huiwe\\Desktop\\20202021\\capstone\\trained agents\\7x7_6P_50N_3L_100S.zip"));
            //newPolicy = DQNPolicy.load("C:\\Users\\huiwe\\Desktop\\20202021\\capstone\\trained agents\\5x5_6P_100N_5L_75S.zip");
        }
        catch(IOException e){
            System.out.println("error!");
        }
        int RLWins = 0;
        int RandomWIns = 0;
        int unfinished = 0;
        ArrayList<Integer> stepsTaken = new ArrayList<Integer>();

        for(int i = 0; i < 100; i++) {
            step = 0;
            while (gameboard.CheckForWin() == 0) {
                System.out.println("move " + step);
                System.out.println("rl agent");
                int action = newPolicy.nextAction(gameboard.state().getData());
                gameboard.RLAgentStep(action);
                gameboard.PrintUserBoard();
                if (gameboard.CheckForWin() == 1) {
                    step++;
                    System.out.println("player one wins");
                    RLWins += 1;
                    stepsTaken.add(step);
                    gameboard.InitiateAll();
                    break;
                }
                step++;
                System.out.println("move " + step);
                System.out.println("random agent");
                gameboard.RandomAgentStep(2);
                gameboard.PrintUserBoard();
                if (gameboard.CheckForWin() == 2) {
                    step++;
                    System.out.println("player two wins");
                    RandomWIns += 1;
                    gameboard.InitiateAll();
                    break;
                }
                if(step > 300){
                    unfinished += 1;
                    gameboard.InitiateAll();
                    break;
                }
                step++;
            }
            step = 0;
        }

        System.out.println("RL Agent Wins: " + RLWins);
        System.out.println("Random Agent Wins: " + RandomWIns);
        System.out.println("Game never ended: " + unfinished);

        double total = 0;
        for(int i = 0; i < stepsTaken.size(); i++){
            total = stepsTaken.get(i) + total;
        }
        double avg = total/stepsTaken.size();
        System.out.println("avg steps taken to win: " + avg);
    }

}






