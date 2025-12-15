class Trie {
    public Trie[] children;
    public int numberOfChildren;
    public String wordValue;

    public Trie() {
        children = new Trie[26];
        numberOfChildren = 0;
        wordValue = null;
    }

    public void add(String word) {
        Trie node = this;
        for (char ch: word.toCharArray()) {
            int i = ch - 'a';
            if (node.children[i] == null) {
                node.children[i] = new Trie();
                node.numberOfChildren++;
            }
            node = node.children[i];
        }
        node.wordValue = word;
    }

    public void remove(String word) {
        Trie parentOfChild = null;
        int childIndexforDelete = 0;
        Trie node = this;
        for (char ch: word.toCharArray()) {
            int i = ch - 'a';
            if (node.children[i] == null) {
                return ;
            }
            if (node.numberOfChildren > 1 || (node.wordValue != null && node.numberOfChildren == 1)) {
                parentOfChild = node;
                childIndexforDelete = i;
            }
            node = node.children[i];
        }
        node.wordValue = null;
        if (node.numberOfChildren == 0 && parentOfChild != null) {
            parentOfChild.children[childIndexforDelete] = null;
            parentOfChild.numberOfChildren--;
        }
    }
}

class Solution {
    private int rowMax;
    private int colMax;
    private int boardMax;
    private char[][] board;
    private Trie root;
    private boolean[][] visited;
    private List<String> output;

    public List<String> findWords(char[][] board, String[] words) {
        output = new ArrayList<String>();
        rowMax = board.length;
        colMax = board[0].length;
        boardMax = rowMax * colMax;
        this.board = board;
        root = new Trie();
        visited = new boolean[rowMax][colMax];

        // Prune obvious false words
        int[] boardCount = new int[26];
        for (char[] row: board) {
            for (char ch: row) {
                boardCount[ch - 'a']++;
            }
        }
        for (String word: words) {
            if (word.length() > boardMax) {
                continue;
            }
            int[] wordCount = new int[26];
            for (char ch: word.toCharArray()) {
                int i = ch - 'a';
                wordCount[i]++;
                if (wordCount[i] > boardCount[i]) {
                    continue;
                }
            }
            root.add(word);
        }

        // dfs
        for (int r = 0; r < rowMax; r++) {
            for (int c = 0; c < colMax; c++) {
                dfs(root, r, c);
            }
        }

        return output;
    }

    public void dfs(Trie node, int r, int c) {
        if (r < 0 || r >= rowMax || c < 0 || c >= colMax || visited[r][c] || node.children[board[r][c] - 'a'] == null) {
            return ;
        }
        node = node.children[board[r][c] - 'a'];
        visited[r][c] = true;
        if (node.wordValue != null) {
            output.add(node.wordValue);
            root.remove(node.wordValue);
        }
        dfs(node, r+1, c);
        dfs(node, r-1, c);
        dfs(node, r, c+1);
        dfs(node, r, c-1);
        visited[r][c] = false;
    }
}