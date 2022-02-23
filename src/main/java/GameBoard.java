package org.deeplearning4j.rl4j.examples.advanced.src.main.java;

import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

class GameBoard {
    int m;
    int n;
    double[][] playerPositions;
    private ArrayList<ArrayList<String>> availableActions = new ArrayList<ArrayList<String>>();
    static String[][] Coordinates;
    static ArrayList<String> possiblejumps = new ArrayList<String>();
    static ArrayList<ArrayList<String>> PathsToSplitPoint = new ArrayList<ArrayList<String>>();
    static ArrayList<String> SplitPoints = new ArrayList<String>();
    static int firstrowHOP = 0;
    static int firstcolHOP = 0;
    static ArrayList<Integer> EvaluatedScores = new ArrayList<Integer>();
    static ArrayList<ArrayList<String>> EvaluatedPaths = new ArrayList<ArrayList<String>>();
    static ArrayList<Integer> FinalScores = new ArrayList<Integer>();
    static ArrayList<ArrayList<String>> FinalPaths = new ArrayList<ArrayList<String>>();
    static Random rnd = new Random(222);
    static double[][] HumanPlayerPositions;



    ArrayList<String> randomMove = new ArrayList<String>();

    GameBoard(int n, int m) {
        // initially, make all actions available
        playerPositions = new double[n][n];
        HumanPlayerPositions = new double[n][n];
        InitializeUserBoard(playerPositions);
        Coordinates = new String[n][n];
        InitializePositionBoard();
        PlacePieces(playerPositions, m);
        availableActions = EnumerateAll(playerPositions, 1);
        this.m = m;
        this.n = n;
    }

    int getAvailableActionsCount() {
        return availableActions.size();
    }

    public double[][] getPlayerPositions(){
        return playerPositions;
    }

    void reset() {
        InitializeUserBoard(playerPositions);
        PlacePieces(playerPositions, m);
        availableActions = EnumerateAll(playerPositions, 1);
    }

    public int CheckForWin() {
        int i = m;
        int y = 0;
        boolean twoWin = true;
        boolean oneWin = true;
        while(y < m){
            for(int x = 0; x < i; x++){
                if(playerPositions[y][x] != 2){
                    twoWin = false;
                }
            }
            i--;
            y++;
        }
        if((m*2)%2==0){
            i = (m);
            y = playerPositions.length-1;
            while(y > (m)){
                for(int x = playerPositions.length-1; x > i; x--){
                    if(playerPositions[y][x] != 1){
                        oneWin = false;
                    }
                }
                y--;
                i++;
            }
        }
        else{
            i = (m);
            y = playerPositions.length-1;
            while(y > (m)){
                for(int x = playerPositions.length-1; x > i; x--){
                    if(playerPositions[y][x] != 1){
                        oneWin = false;
                    }
                }
                y--;
                i++;
            }
        }
        if(oneWin == true){
            return 1;
        }
        if(twoWin == true){
            return 2;
        }
        else{
            return 0;
        }
    }


    public ArrayList<String> RLAgentStep(int action){
        ArrayList<String> ChosenMove = availableActions.get(action);
        MovePiece(ChosenMove.get(0), ChosenMove.get(ChosenMove.size()-1), playerPositions);
        availableActions = EnumerateAll(playerPositions, 1);
        return ChosenMove;
    }

    public ArrayList<String> GreedyAgentStep(int player){
        ArrayList<String> greedyMove = chooseGreedyAction(EnumerateAll(playerPositions, player), player, playerPositions);
        playerPositions = MovePiece(greedyMove.get(0), greedyMove.get(greedyMove.size() - 1), playerPositions);
        availableActions = EnumerateAll(playerPositions, 1);
        return greedyMove;
    }

    public ArrayList<String> RandomAgentStep(int player){
        ArrayList<String> randomMove = chooseRandomMove(EnumerateAll(playerPositions, player));
        playerPositions = MovePiece(randomMove.get(0), randomMove.get(randomMove.size() - 1), playerPositions);
        availableActions = EnumerateAll(playerPositions, 1);
        return randomMove;
    }
    State state() { //return enviroment here
        ArrayList<double[]> actionStatePairs = new ArrayList<double[]>();
        double[][] CopyOfplayerPositions = new double[n][n];
        // for each action, add the corresponding state-action representation
        for (int j = 0; j < availableActions.size(); j++) {
            for (int i = 0; i < playerPositions.length; i++) {
                for (int l = 0; l < playerPositions.length; l++) {
                    CopyOfplayerPositions[i][l] = playerPositions[i][l];
                }
            }
            actionStatePairs.add(twoToOne(MovePiece(availableActions.get(j).get(0), availableActions.get(j).get(availableActions.get(j).size() - 1), CopyOfplayerPositions), availableActions.get(j)));

        }
        return new State(actionStatePairs);
    }
    public void InitiateAll(){
        InitializeUserBoard(playerPositions);
        InitializePositionBoard();
        PlacePieces(playerPositions, m);
    }
    public void PrintUserBoard(){
        for(int i = 0; i < playerPositions.length; i++){
            for (int j = 0; j < playerPositions.length; j++){
                System.out.print(playerPositions[i][j]);
                System.out.print(" ");
            }
            System.out.println();
        }
    }

    public static double[][] MovePiece(String start, String end, double[][] grid) {
        double Piece = 0;
        for (int z = 0; z < Coordinates.length; z++) {
            for (int x = 0; x < Coordinates.length; x++) {
                if (Coordinates[z][x] == start) {
                    Piece = grid[z][x];
                    grid[z][x] = 0;
                }
            }
        }
        for (int a = 0; a < Coordinates.length; a++) {
            for (int b = 0; b < Coordinates.length; b++) {
                if (Coordinates[a][b] == end) {
                    grid[a][b] = Piece;
                }
            }
        }
        return grid;
    }
    public static double[] twoToOne(double[][] grid, ArrayList<String> path) {
        double[] array = new double[(grid.length * grid.length) + 3];

        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid.length; j++) {
                array[i * grid.length + j] = (grid[i][j] == 2.0)? -1.0: grid[i][j];
            }
        }
        //append score to end
        double score = EvaluatePath(path);

        array[array.length-3] = score;

        //append avg manhatan distance for both players
        array[array.length-2] = avgDistance(grid, 1);
        array[array.length-1] = avgDistance(grid, 2);
        return array;
    }
    public static void InitializeUserBoard(double[][] grid){
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid.length; j++) {
                grid[i][j] = 0;
            }
        }
    }
    public static void InitializePositionBoard() {
        String alphabet = "abcdefghijklmnopzrstuvwxyz";
        for (int i = 0; i < Coordinates.length; i++) {
            for (int j = 0; j < Coordinates.length; j++) {
                String number = Integer.toString(j);
                Coordinates[i][j] = alphabet.charAt(i) + number;
            }
        }
    }
    public static void PlacePieces(double[][] grid, int m){
        int i = m;
        int y = 0;
        while(y < m){
            for(int x = 0; x < i; x++){
                grid[y][x] = 1;
            }
            i--;
            y++;
        }
        if((m*2)%2==0){
            i = (m);
            y = grid.length-1;
            while(y > (m)){
                for(int x = grid.length-1; x > i; x--){
                    grid[y][x] = 2;
                }
                y--;
                i++;
            }
        }
        else{
            i = (m);
            y = grid.length-1;
            while(y > (m)){
                for(int x = grid.length-1; x > i; x--){
                    grid[y][x] = 2;
                }
                y--;
                i++;
            }
        }
    }

    public static void DFS(double[][] grid, int startrow, int startcol, ArrayList<ArrayList<String>> EnumPaths) {
        int h = grid.length;
        if (h == 0)
            return;
        int l = grid[0].length;

        //created visited array
        boolean [][] visited = new boolean[h][l];
        firstrowHOP = startrow;
        firstcolHOP = firstcolHOP;

        ArrayList<String> Path =  new ArrayList<String>();
        DFSUtil(grid, Coordinates, startrow, startcol, visited, Path, EnumPaths);
    }
    public static void DFSUtil(double[][] grid, String[][] Coordinates, int row, int col, boolean[][] visited, ArrayList<String> Path, ArrayList<ArrayList<String>> EnumPaths) {
        int H = grid.length;
        int L = grid[0].length;

        if(visited[row][col] == true){
            return;
        }

        //mark the cell visited
        visited[row][col] = true;
        Path.clear();

        //single hop
        if(row + 1 < grid.length && grid[row+1][col]==0 || row - 1 > 0 && grid[row-1][col]==0 || col + 1 < grid.length && grid[row][col+1]==0 || col - 1 > 0 && grid[row][col-1]==0 || row + 1 < grid.length && col + 1 < grid.length && grid[row+1][col+1]==0 || row - 1 > 0 && col-1 > 0 && grid[row-1][col-1]==0){
            if(row + 1 < grid.length && grid[row+1][col]==0){
                Path.add(Coordinates[row][col]);
                Path.add(Coordinates[row+1][col]);
                EnumPaths.add(new ArrayList<>(Path));
                Path.clear();
            }

            if(row - 1 >= 0 && grid[row-1][col]==0){
                Path.add(Coordinates[row][col]);
                Path.add(Coordinates[row-1][col]);
                EnumPaths.add(new ArrayList<>(Path));
                Path.clear();
            }

            if(col + 1 < grid.length && grid[row][col+1]==0){
                Path.add(Coordinates[row][col]);
                Path.add(Coordinates[row][col+1]);
                EnumPaths.add(new ArrayList<>(Path));
                Path.clear();
            }

            if(col - 1 >= 0 && grid[row][col-1]==0){
                Path.add(Coordinates[row][col]);
                Path.add(Coordinates[row][col-1]);
                EnumPaths.add(new ArrayList<>(Path));
                Path.clear();
            }

            if(row + 1 < grid.length && col - 1 >= 0 && grid[row+1][col-1]==0){
                Path.add(Coordinates[row][col]);
                Path.add(Coordinates[row+1][col-1]);
                EnumPaths.add(new ArrayList<>(Path));
                Path.clear();
            }

            if(row - 1 >= 0 && col + 1 < grid.length && grid[row-1][col+1]==0){
                Path.add(Coordinates[row][col]);
                Path.add(Coordinates[row-1][col+1]);
                EnumPaths.add(new ArrayList<>(Path));
                Path.clear();
            }
        }
        //multiple jumps
        if(row+2 < grid.length && grid[row+1][col]==1 && grid[row+2][col]==0 && visited[row+2][col] == false || row+2 < grid.length && grid[row+1][col]==2 && grid[row+2][col]==0 && visited[row+2][col] == false){
            Path.add(Coordinates[firstrowHOP][firstcolHOP]);
            Path.add(Coordinates[row+2][col]);
            EnumPaths.add(new ArrayList<>(Path));
            Path.clear();
            DFSUtilHop(grid, Coordinates, row+2, col, visited, Path, EnumPaths);
        }
        if(row-2 >= 0 && grid[row-1][col]==1 && grid[row-2][col]==0 && visited[row-2][col] == false || row-2 >= 0 && grid[row-1][col]==2 && grid[row-2][col]==0 && visited[row-2][col] == false){
            Path.add(Coordinates[firstrowHOP][firstcolHOP]);
            Path.add(Coordinates[row-2][col]);
            EnumPaths.add(new ArrayList<>(Path));
            Path.clear();
            DFSUtilHop(grid, Coordinates, row-2, col, visited, Path, EnumPaths);
        }
        if(col+2 < grid.length && grid[row][col+1]==1 && grid[row][col+2]==0 && visited[row][col+2] == false || col+2 < grid.length && grid[row][col+1]==2 && grid[row][col+2]==0 && visited[row][col+2] == false){
            Path.add(Coordinates[firstrowHOP][firstcolHOP]);
            Path.add(Coordinates[row][col+2]);
            EnumPaths.add(new ArrayList<>(Path));
            Path.clear();
            DFSUtilHop(grid, Coordinates, row, col+2, visited, Path, EnumPaths);
        }
        if(col-2 >= 0 && grid[row][col-1]==1 && grid[row][col-2]==0 && visited[row][col-2] == false || col-2 >= 0 && grid[row][col-1]==2 && grid[row][col-2]==0 && visited[row][col-2] == false){
            Path.add(Coordinates[firstrowHOP][firstcolHOP]);
            Path.add(Coordinates[row][col-2]);
            EnumPaths.add(new ArrayList<>(Path));
            Path.clear();
            DFSUtilHop(grid, Coordinates, row, col-2, visited, Path, EnumPaths);
        }
        if(col-2 >= 0 && row+2 < grid.length && grid[row+1][col-1]==1 && grid[row+2][col-2]==0 && visited[row+2][col-2] == false || col-2 >= 0 && row + 2 < grid.length && grid[row+1][col-1]==2 && grid[row+2][col-2]==0 && visited[row+2][col-2] == false){
            Path.add(Coordinates[firstrowHOP][firstcolHOP]);
            Path.add(Coordinates[row+2][col-2]);
            EnumPaths.add(new ArrayList<>(Path));
            Path.clear();
            DFSUtilHop(grid, Coordinates, row+2, col-2, visited, Path, EnumPaths);
        }
        if(col+2 < grid.length && row-2 >= 0 && grid[row-1][col+1]==1 && grid[row-2][col+2]==0 && visited[row-2][col+2] == false || col+2 < grid.length && row - 2 >= 0 && grid[row-1][col+1]==2 && grid[row-2][col+2]==0 && visited[row-2][col+2] == false){
            Path.add(Coordinates[firstrowHOP][firstcolHOP]);
            Path.add(Coordinates[row-2][col+2]);
            EnumPaths.add(new ArrayList<>(Path));
            Path.clear();
            DFSUtilHop(grid, Coordinates, row-2, col+2, visited, Path, EnumPaths);
        }
        else{
        }
    }
    //DFS for hopping [CPU]
    public static void DFSUtilHop(double[][] grid, String[][] Coordinates, int row, int col, boolean[][] visited, ArrayList<String> Path, ArrayList<ArrayList<String>> EnumPaths){
        int H = grid.length;
        int L = grid[0].length;

        possiblejumps.clear();

        if (visited[row][col] == true){
            return;
        }

        visited[row][col] = true;

        if(row+2 < grid.length && grid[row+1][col]==1 && grid[row+2][col]==0 && visited[row+2][col] == false || row+2 < grid.length && grid[row+1][col]==2 && grid[row+2][col]==0 && visited[row+2][col] == false){
            Path.add(Coordinates[firstrowHOP][firstcolHOP]);
            Path.add(Coordinates[row+2][col]);
            EnumPaths.add(new ArrayList<>(Path));
            Path.clear();
            DFSUtilHop(grid, Coordinates, row+2, col, visited, Path, EnumPaths);
        }

        if(row-2 >= 0 && grid[row-1][col]==1 && grid[row-2][col]==0 && visited[row-2][col] == false || row-2 >= 0 && grid[row-1][col]==2 && grid[row-2][col]==0 && visited[row-2][col] == false){
            Path.add(Coordinates[firstrowHOP][firstcolHOP]);
            Path.add(Coordinates[row-2][col]);
            EnumPaths.add(new ArrayList<>(Path));
            Path.clear();
            DFSUtilHop(grid, Coordinates, row-2, col, visited, Path, EnumPaths);
        }

        if(col+2 < grid.length && grid[row][col+1]==1 && grid[row][col+2]==0 && visited[row][col+2] == false|| col+2 < grid.length && grid[row][col+1]==2 && grid[row][col+2]==0 && visited[row][col+2] == false){
            Path.add(Coordinates[firstrowHOP][firstcolHOP]);
            Path.add(Coordinates[row][col+2]);
            EnumPaths.add(new ArrayList<>(Path));
            Path.clear();
            DFSUtilHop(grid, Coordinates, row, col+2, visited, Path, EnumPaths);
        }

        if(col-2 >= 0 && grid[row][col-1]==1 && grid[row][col-2]==0 && visited[row][col-2] == false|| col-2 >= 0 && grid[row][col-1]==2 && grid[row][col-2]==0 && visited[row][col-2] == false){
            Path.add(Coordinates[firstrowHOP][firstcolHOP]);
            Path.add(Coordinates[row][col-2]);
            EnumPaths.add(new ArrayList<>(Path));
            Path.clear();
            DFSUtilHop(grid, Coordinates, row, col-2, visited, Path, EnumPaths);
        }

        if(col-2 >= 0 && row+2 < grid.length && grid[row+1][col-1]==1 && grid[row+2][col-2]==0 && visited[row+2][col-2] == false|| col-2 >= 0 && row + 2 < grid.length && grid[row+1][col-1]==2 && grid[row+2][col-2]==0 && visited[row+2][col-2] == false){
            Path.add(Coordinates[firstrowHOP][firstcolHOP]);
            Path.add(Coordinates[row+2][col-2]);
            EnumPaths.add(new ArrayList<>(Path));
            Path.clear();
            DFSUtilHop(grid, Coordinates, row+2, col-2, visited, Path, EnumPaths);

        }

        if(col+2 < grid.length && row-2 >= 0 && grid[row-1][col+1]==1 && grid[row-2][col+2]==0 && visited[row-2][col+2] == false || col+2 < grid.length && row - 2 >= 0 && grid[row-1][col+1]==2 && grid[row-2][col+2]==0 && visited[row-2][col+2] == false){
            Path.add(Coordinates[firstrowHOP][firstcolHOP]);
            Path.add(Coordinates[row-2][col+2]);
            EnumPaths.add(new ArrayList<>(Path));
            Path.clear();
            DFSUtilHop(grid, Coordinates, row-2, col+2, visited, Path, EnumPaths);
        }
        else{
            DFSUtil(grid, Coordinates, row, col, visited, Path, EnumPaths);
        }

    }
    //LOOPS THROUGH ALL PIECES ON BOARD AND ENUMERATES POSSIBLE PATHS FOR ALL OF THEM
    public static ArrayList<ArrayList<String>> EnumerateAll(double[][] playerPositions, int Piece) {
        ArrayList<ArrayList<String>> EnumPaths = new ArrayList<ArrayList<String>>();
        for (int i = 0; i < playerPositions.length; i++) {
            for (int j = 0; j < playerPositions.length; j++) {
                if (Piece == 1 && playerPositions[i][j] == 1) {
                    firstrowHOP = i;
                    firstcolHOP = j;
                    DFS(playerPositions, i, j, EnumPaths);
                }
                if (Piece == 2 && playerPositions[i][j] == 2) {
                    firstrowHOP = i;
                    firstcolHOP = j;
                    DFS(playerPositions, i, j, EnumPaths);
                }
            }
        }
        ArrayList<ArrayList<String>> EnumPathsFinal = new ArrayList<ArrayList<String>>();
        for (int i = 0; i < EnumPaths.size(); i++) {
            if (EnumPaths.get(i).size() != 1 && EnumPaths.get(i).size() != 0) {
                EnumPathsFinal.add(EnumPaths.get(i));
            }
        }
        return EnumPathsFinal;
    }
    public static double[][] GreedyAction(ArrayList<ArrayList<String>> EnumPaths, int PlayerNumber, double[][] grid) {
        int MaxIndex = 0;
        EvaluatedScores.clear();
        EvaluatedPaths.clear();
        FinalScores.clear();
        FinalPaths.clear();

        for (int i = 0; i < EnumPaths.size(); i++) {
            if (EnumPaths.get(i).size() < 2) {
                continue;
            }

            String StartPosition = EnumPaths.get(i).get(0);
            String EndPosition = EnumPaths.get(i).get(EnumPaths.get(i).size() - 1);

            for (int z = 0; z < Coordinates.length; z++) {
                for (int x = 0; x < Coordinates.length; x++) {
                    if (StartPosition == Coordinates[z][x] && grid[z][x] == PlayerNumber && PlayerNumber == 1) {
                        for (int q = 0; q < Coordinates.length; q++) {
                            for (int j = 0; j < Coordinates.length; j++) {
                                if (EndPosition == Coordinates[q][j]) {
                                    EvaluatedScores.add((q - z) + (j - x));
                                    EvaluatedPaths.add(EnumPaths.get(i));
                                }
                            }
                        }
                    }
                    if (StartPosition == Coordinates[z][x] && grid[z][x] == PlayerNumber && PlayerNumber == 2) {
                        for (int a = 0; a < Coordinates.length; a++) {
                            for (int b = 0; b < Coordinates.length; b++) {
                                if (EndPosition == Coordinates[a][b]) {
                                    EvaluatedScores.add((a - z) + (b - x));
                                    EvaluatedPaths.add(EnumPaths.get(i));
                                }
                            }
                        }
                    }
                }
            }
        }
        double CurrentHighest = -100;
        double CurrentLowest = 100;
        for (int y = 0; y < EvaluatedScores.size(); y++) {
            if (PlayerNumber == 1) {
                if (CurrentHighest < EvaluatedScores.get(y)) {
                    FinalPaths.clear();
                    FinalScores.clear();
                    FinalScores.add(EvaluatedScores.get(y));
                    FinalPaths.add(EvaluatedPaths.get(y));
                    CurrentHighest = EvaluatedScores.get(y);
                }
                if (CurrentHighest == EvaluatedScores.get(y)) {
                    FinalScores.add(EvaluatedScores.get(y));
                    FinalPaths.add(EvaluatedPaths.get(y));
                }
            }
            if (PlayerNumber == 2) {
                if (CurrentLowest > EvaluatedScores.get(y)) {
                    FinalPaths.clear();
                    FinalScores.clear();
                    FinalScores.add(EvaluatedScores.get(y));
                    FinalPaths.add(EvaluatedPaths.get(y));
                    CurrentLowest = EvaluatedScores.get(y);
                }
                if (CurrentLowest == EvaluatedScores.get(y)) {
                    FinalScores.add(EvaluatedScores.get(y));
                    FinalPaths.add(EvaluatedPaths.get(y));
                }
            }
        }

        int i = rnd.nextInt(FinalScores.size());
        ArrayList<String> BestPath = new ArrayList<String>();
        BestPath = FinalPaths.get(i);
        return MovePiece(BestPath.get(0), BestPath.get(BestPath.size() - 1), grid);
    }
    public static ArrayList<String> chooseGreedyAction(ArrayList<ArrayList<String>> EnumPaths, int PlayerNumber, double[][] grid) {
        int MaxIndex = 0;
        EvaluatedScores.clear();
        EvaluatedPaths.clear();
        FinalScores.clear();
        FinalPaths.clear();

        for (int i = 0; i < EnumPaths.size(); i++) {
            if (EnumPaths.get(i).size() < 2) {
                continue;
            }

            String StartPosition = EnumPaths.get(i).get(0);
            String EndPosition = EnumPaths.get(i).get(EnumPaths.get(i).size() - 1);

            for (int z = 0; z < Coordinates.length; z++) {
                for (int x = 0; x < Coordinates.length; x++) {
                    if (StartPosition == Coordinates[z][x] && grid[z][x] == PlayerNumber && PlayerNumber == 1) {
                        for (int q = 0; q < Coordinates.length; q++) {
                            for (int j = 0; j < Coordinates.length; j++) {
                                if (EndPosition == Coordinates[q][j]) {
                                    EvaluatedScores.add((q - z) + (j - x));
                                    EvaluatedPaths.add(EnumPaths.get(i));
                                }
                            }
                        }
                    }
                    if (StartPosition == Coordinates[z][x] && grid[z][x] == PlayerNumber && PlayerNumber == 2) {
                        for (int a = 0; a < Coordinates.length; a++) {
                            for (int b = 0; b < Coordinates.length; b++) {
                                if (EndPosition == Coordinates[a][b]) {
                                    EvaluatedScores.add((a - z) + (b - x));
                                    EvaluatedPaths.add(EnumPaths.get(i));
                                }
                            }
                        }
                    }
                }
            }
        }
        double CurrentHighest = -100;
        double CurrentLowest = 100;
        for (int y = 0; y < EvaluatedScores.size(); y++) {
            if (PlayerNumber == 1) {
                if (CurrentHighest < EvaluatedScores.get(y)) {
                    FinalPaths.clear();
                    FinalScores.clear();
                    FinalScores.add(EvaluatedScores.get(y));
                    FinalPaths.add(EvaluatedPaths.get(y));
                    CurrentHighest = EvaluatedScores.get(y);
                }
                if (CurrentHighest == EvaluatedScores.get(y)) {
                    FinalScores.add(EvaluatedScores.get(y));
                    FinalPaths.add(EvaluatedPaths.get(y));
                }
            }
            if (PlayerNumber == 2) {
                if (CurrentLowest > EvaluatedScores.get(y)) {
                    FinalPaths.clear();
                    FinalScores.clear();
                    FinalScores.add(EvaluatedScores.get(y));
                    FinalPaths.add(EvaluatedPaths.get(y));
                    CurrentLowest = EvaluatedScores.get(y);
                }
                if (CurrentLowest == EvaluatedScores.get(y)) {
                    FinalScores.add(EvaluatedScores.get(y));
                    FinalPaths.add(EvaluatedPaths.get(y));
                }
            }
        }

        int i = rnd.nextInt(FinalScores.size());
        ArrayList<String> BestPath = new ArrayList<String>();
        BestPath = FinalPaths.get(i);
        return BestPath;
    }
    public static double[][] RandomMove(ArrayList<ArrayList<String>> EnumPaths, double[][] grid) {
        EvaluatedScores.clear();
        EvaluatedPaths.clear();
        FinalScores.clear();
        FinalPaths.clear();

        for (int i = 0; i < EnumPaths.size(); i++) {
            if (EnumPaths.get(i).size() < 2) {
                continue;
            }

            String StartPosition = EnumPaths.get(i).get(0);
            String EndPosition = EnumPaths.get(i).get(EnumPaths.get(i).size() - 1);

            for (int z = 0; z < Coordinates.length; z++) {
                for (int x = 0; x < Coordinates.length; x++) {
                    if (StartPosition == Coordinates[z][x]) {
                        for (int a = 0; a < Coordinates.length; a++) {
                            for (int b = 0; b < Coordinates.length; b++) {
                                if (EndPosition == Coordinates[a][b]) {
                                    EvaluatedScores.add((a - z) + (b - x));
                                    EvaluatedPaths.add(EnumPaths.get(i));
                                }
                            }
                        }
                    }
                }
            }
        }
        for (int i = 0; i < EvaluatedScores.size() - 1; i++) {
            if (EvaluatedScores.get(i) >= 0) {
                EvaluatedScores.remove(i);
                EvaluatedPaths.remove(i);
            }
        }
        int RANDOM = rnd.nextInt(EvaluatedPaths.size());
        ArrayList<String> BestPath = new ArrayList<String>();
        BestPath = EvaluatedPaths.get(RANDOM);
        return MovePiece(BestPath.get(0), BestPath.get(BestPath.size()-1), grid);
    }
    public static ArrayList<String> chooseRandomMove(ArrayList<ArrayList<String>> EnumPaths) {
        EvaluatedScores.clear();
        EvaluatedPaths.clear();
        FinalScores.clear();
        FinalPaths.clear();

        for (int i = 0; i < EnumPaths.size(); i++) {
            if (EnumPaths.get(i).size() < 2) {
                continue;
            }

            String StartPosition = EnumPaths.get(i).get(0);
            String EndPosition = EnumPaths.get(i).get(EnumPaths.get(i).size() - 1);

            for (int z = 0; z < Coordinates.length; z++) {
                for (int x = 0; x < Coordinates.length; x++) {
                    if (StartPosition == Coordinates[z][x]) {
                        for (int a = 0; a < Coordinates.length; a++) {
                            for (int b = 0; b < Coordinates.length; b++) {
                                if (EndPosition == Coordinates[a][b]) {
                                    EvaluatedScores.add((a - z) + (b - x));
                                    EvaluatedPaths.add(EnumPaths.get(i));
                                }
                            }
                        }
                    }
                }
            }
        }
        for (int i = 0; i < EvaluatedScores.size() - 1; i++) {
            if (EvaluatedScores.get(i) >= 0) {
                EvaluatedScores.remove(i);
                EvaluatedPaths.remove(i);
            }
        }
        int RANDOM = rnd.nextInt(EvaluatedPaths.size());
        ArrayList<String> BestPath = new ArrayList<String>();
        BestPath = EvaluatedPaths.get(RANDOM);
        return BestPath;
    }
    public static int EvaluatePath(ArrayList<String> EnumPaths) {
        int score = 0;

        String StartPosition = EnumPaths.get(0);
        String EndPosition = EnumPaths.get(EnumPaths.size() - 1);

        for (int z = 0; z < Coordinates.length; z++) {
            for (int x = 0; x < Coordinates.length; x++) {
                if (StartPosition == Coordinates[z][x]) {
                    for (int q = 0; q < Coordinates.length; q++) {
                        for (int j = 0; j < Coordinates.length; j++) {
                            if (EndPosition == Coordinates[q][j]) {
                                score = ((q - z) + (j - x));
                            }
                        }
                    }
                }
            }
        }
        return score;
    }
    //DFS [HUMAN]
    public static void HumanDFS(double[][] grid, int startrow, int startcol) {
        int h = grid.length;
        if (h == 0)
            return;
        int l = grid[0].length;
        //created visited array
        boolean [][] visited = new boolean[h][l];
        HumanDFSUtil(grid, Coordinates, startrow,  startcol, visited);
    }
    public void UserInput2(){
        for(int i = 0; i < playerPositions.length; i++){
            for(int j = 0; j < playerPositions.length; j++){
                HumanPlayerPositions[i][j] = playerPositions[i][j];
            }
        }
        System.out.println("choose a piece to move");
        Scanner in = new Scanner(System.in);
        String choice = in.nextLine();
        for(int a = 0; a < Coordinates.length; a++){
            for(int b = 0; b < Coordinates.length; b++){
                if(Coordinates[a][b].equals(choice)){
                    if(playerPositions[a][b] == 2){
                        System.out.println("choose a place to move");
                        HumanDFS(HumanPlayerPositions, a, b);
                        for(int i = 0; i < playerPositions.length; i++){
                            for(int j = 0; j < playerPositions.length; j++){
                                System.out.print(HumanPlayerPositions[i][j]  + " ");
                            }
                            System.out.println("");
                        }
                        String choice2 = in.nextLine();
                        for(int x = 0; x < playerPositions.length; x++){
                            for(int y = 0; y < playerPositions.length; y++){
                                if(Coordinates[x][y].equals(choice2)){
                                    if(HumanPlayerPositions[x][y] == 3){
                                        playerPositions[a][b] = 0;
                                        playerPositions[x][y] = 2;
                                    }
                                    else{
                                        System.out.println("Invalid input1");
                                        PrintUserBoard();
                                        UserInput2();
                                    }
                                }
                            }
                        }
                    }
                    else{
                        System.out.println("Invalid Input");
                        UserInput2();
                    }
                }
            }
        }
    }

    public static void HumanDFSUtil(double[][] grid, String[][] Coordinates, int row, int col, boolean[][] visited) {
        int H = grid.length;
        int L = grid[0].length;
        if(row < 0 || col < 0 || row >= H || col >= L || visited[row][col] == true){
            return;
        }

        //mark the cell visited
        visited[row][col] = true;

        //single hop
        if(row + 1 < grid.length && grid[row+1][col]==0 && visited[row+1][col] == false){
            HumanPlayerPositions[row+1][col] = 3;
            HumanDFSUtil(grid, Coordinates, row, col, visited);
        }

        if(row - 1 > 0 && grid[row-1][col]==0 && visited[row-1][col] == false){
            HumanPlayerPositions[row-1][col] = 3;;
            HumanDFSUtil(grid, Coordinates, row, col, visited);
        }

        if(col + 1 < grid.length && grid[row][col+1]==0 && visited[row][col+1] == false){
            HumanPlayerPositions[row][col+1] = 3;
            HumanDFSUtil(grid, Coordinates, row, col, visited);
        }

        if(col - 1 > 0 && grid[row][col-1]==0 && visited[row][col-1] == false){
            HumanPlayerPositions[row][col-1] = 3;
            HumanDFSUtil(grid, Coordinates, row, col, visited);
        }

        if(row + 1 < grid.length && col - 1 >= 0 && grid[row+1][col-1]==0){
            HumanPlayerPositions[row+1][col-1] = 3;
            HumanDFSUtil(grid, Coordinates, row, col, visited);
        }

        if(row - 1 >= 0 && col + 1 < grid.length && grid[row-1][col+1]==0){
            HumanPlayerPositions[row-1][col+1] = 3;
            HumanDFSUtil(grid, Coordinates, row, col, visited);
        }

        //multiple jumps
        if(row+2 < grid.length && grid[row+1][col]==1 && grid[row+2][col]==0 || row+2 < grid.length && grid[row+1][col]==2 && grid[row+2][col]==0 && visited[row+2][col] == false){
            HumanPlayerPositions[row+2][col] = 3;
            HumanDFSUtilHop(grid, Coordinates, row+2, col, visited);
        }

        if(row-2 >= 0 && grid[row-1][col]==1 && grid[row-2][col]==0 || row-2 > 0 && grid[row-1][col]==2 && grid[row-2][col]==0 && visited[row-2][col] == false){
            HumanPlayerPositions[row-2][col] = 3;
            HumanDFSUtilHop(grid, Coordinates, row-2, col, visited);
        }

        if(col+2 < grid.length && grid[row][col+1]==1 && grid[row][col+2]==0 || col+2 < grid.length && grid[row][col+1]==2 && grid[row][col+2]==0 && visited[row][col+2] == false){
            HumanPlayerPositions[row][col+2] = 3;
            HumanDFSUtilHop(grid, Coordinates, row, col+2, visited);
        }

        if(col-2 >= 0 && grid[row][col-2]==1 && grid[row][col-2]==0 || col-2 > 0 && grid[row][col-1]==2 && grid[row][col-2]==0 && visited[row][col-2] == false){
            HumanPlayerPositions[row][col-2] = 3;
            HumanDFSUtilHop(grid, Coordinates, row, col-2, visited);
        }

        if(col-2 >= 0 && row+2 < grid.length && grid[row+1][col-1]==1 && grid[row+2][col-2]==0 && visited[row+2][col-2] == false|| col-2 >= 0 && row + 2 < grid.length && grid[row+1][col-1]==2 && grid[row+2][col-2]==0 && visited[row+2][col-2] == false){
            HumanPlayerPositions[row+2][col-2] = 3;
            HumanDFSUtilHop(grid, Coordinates, row+2, col-2, visited);
        }

        if(col+2 < grid.length && row-2 >= 0 && grid[row-1][col+1]==1 && grid[row-2][col+2]==0 && visited[row-2][col+2] == false || col+2 < grid.length && row - 2 >= 0 && grid[row-1][col+1]==2 && grid[row-2][col+2]==0 && visited[row-2][col+2] == false){
            HumanPlayerPositions[row-2][col+2] = 3;
            HumanDFSUtilHop(grid, Coordinates, row-2, col-2, visited);
        }
    }

    //DFS for hopping [HUMAN]
    public static void HumanDFSUtilHop(double[][] grid, String[][] Coordinates, int row, int col, boolean[][] visited){
        int H = grid.length;
        int L = grid[0].length;
        if (row < 0 || col < 0 || row >= H || col >= L || visited[row][col] == true){
            return;
        }
        visited[row][col] = true;
        HumanPlayerPositions[row][col] = 3;

        if(row+2 < grid.length && grid[row+1][col]==1 && grid[row+2][col]==0 || row+2 < grid.length && grid[row+1][col]==2 && grid[row+2][col]==0){
            HumanDFSUtilHop(grid, Coordinates, row+2, col, visited);
        }

        if(row-2 >= 0 && grid[row-1][col]==1 && grid[row-2][col]==0 || row-2 > 0 && grid[row-1][col]==2 && grid[row-2][col]==0){
            HumanDFSUtilHop(grid, Coordinates, row-2, col, visited);
        }

        if(col+2 < grid.length && grid[row][col+1]==1 && grid[row][col+2]==0 || col+2 < grid.length && grid[row][col+1]==2 && grid[row][col+2]==0){
            HumanDFSUtilHop(grid, Coordinates, row, col+2, visited);
        }

        if(col-2 >= 0 && grid[row][col-1]==1 && grid[row][col-2]==0 || col-2 > 0 && grid[row][col-1]==2 && grid[row][col-2]==0){
            HumanDFSUtilHop(grid, Coordinates, row, col-2, visited);
        }

        if(col-2 >= 0 && row+2 < grid.length && grid[row+1][col-1]==1 && grid[row+2][col-2]==0 && visited[row+2][col-2] == false|| col-2 >= 0 && row + 2 < grid.length && grid[row+1][col-1]==2 && grid[row+2][col-2]==0 && visited[row+2][col-2] == false){
            HumanPlayerPositions[row+2][col-2] = 3;
            HumanDFSUtilHop(grid, Coordinates, row+2, col-2, visited);
        }

        if(col+2 < grid.length && row-2 >= 0 && grid[row-1][col+1]==1 && grid[row-2][col+2]==0 && visited[row-2][col+2] == false || col+2 < grid.length && row - 2 >= 0 && grid[row-1][col+1]==2 && grid[row-2][col+2]==0 && visited[row-2][col+2] == false){
            HumanPlayerPositions[row-2][col+2] = 3;
            HumanDFSUtilHop(grid, Coordinates, row-2, col-2, visited);
        }

        else{
            HumanDFSUtil(grid, Coordinates, row, col, visited);
        }
    }
    public static double avgDistance(double[][]grid, int piece){
        int total = 0;
        int pieces = 0;
        if(piece == 1) {
            for (int i = 0; i < grid.length; i++) {
                for (int j = 0; j < grid.length; j++) {
                    if (grid[i][j] == 1.0) {
                        total += i + j;
                        pieces++;
                    }
                }
            }
        }
        if(piece == 2){
            for (int i = 0; i < grid.length; i++) {
                for (int j = 0; j < grid.length; j++) {
                    if (grid[i][j] == 2.0) {
                        total += 2 * (grid.length - 1)  - (i + j);
                        pieces++;
                    }
                }
            }
        }
        return total * 1.0 / pieces;
    }

}
