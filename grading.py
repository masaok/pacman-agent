from environment import *
from maze_clause import *
from maze_knowledge_base import *
from copy import deepcopy
import timeout_decorator

class MazeKBTests(unittest.TestCase):
    
    # -----------------------------------------------------------------------------------------
    # MazeClause Tests
    # -----------------------------------------------------------------------------------------
    def test_mazeprops1(self):
        mc = MazeClause([(("X", (1, 1)), True), (("X", (2, 1)), True), (("Y", (1, 2)), False)])
        self.assertTrue(mc.get_prop(("X", (1, 1))))
        self.assertTrue(mc.get_prop(("X", (2, 1))))
        self.assertFalse(mc.get_prop(("Y", (1, 2))))
        self.assertTrue(mc.get_prop(("X", (2, 2))) is None)
        self.assertFalse(mc.is_empty())
        
    def test_mazeprops2(self):
        mc = MazeClause([(("X", (1, 1)), True), (("X", (1, 1)), True)])
        self.assertTrue(mc.get_prop(("X", (1, 1))))
        self.assertFalse(mc.is_empty())
        
    def test_mazeprops3(self):
        mc = MazeClause([(("X", (1, 1)), True), (("Y", (2, 1)), True), (("X", (1, 1)), False)])
        self.assertTrue(mc.is_valid())
        self.assertTrue(mc.get_prop(("X", (1, 1))) is None)
        self.assertFalse(mc.is_empty())
        
    def test_mazeprops4(self):
        mc = MazeClause([])
        self.assertFalse(mc.is_valid())
        self.assertTrue(mc.is_empty())
        
    def test_mazeprops5(self):
        mc1 = MazeClause([(("X", (1, 1)), True)])
        mc2 = MazeClause([(("X", (1, 1)), True)])
        res = MazeClause.resolve(mc1, mc2)
        self.assertEqual(len(res), 0)
        
    def test_mazeprops6(self):
        mc1 = MazeClause([(("X", (1, 1)), True)])
        mc2 = MazeClause([(("X", (1, 1)), False)])
        res = MazeClause.resolve(mc1, mc2)
        self.assertEqual(len(res), 1)
        self.assertTrue(MazeClause([]) in res)
        
    def test_mazeprops7(self):
        mc1 = MazeClause([(("X", (1, 1)), True), (("Y", (1, 1)), True)])
        mc2 = MazeClause([(("X", (1, 1)), False), (("Y", (2, 2)), True)])
        res = MazeClause.resolve(mc1, mc2)
        self.assertEqual(len(res), 1)
        self.assertTrue(MazeClause([(("Y", (1, 1)), True), (("Y", (2, 2)), True)]) in res)
        
    def test_mazeprops8(self):
        mc1 = MazeClause([(("X", (1, 1)), True), (("Y", (1, 1)), False)])
        mc2 = MazeClause([(("X", (1, 1)), False), (("Y", (1, 1)), True)])
        res = MazeClause.resolve(mc1, mc2)
        self.assertEqual(len(res), 0)
        
    def test_mazeprops9(self):
        mc1 = MazeClause([(("X", (1, 1)), True), (("Y", (1, 1)), False), (("Z", (1, 1)), True)])
        mc2 = MazeClause([(("X", (1, 1)), False), (("Y", (1, 1)), True), (("W", (1, 1)), False)])
        res = MazeClause.resolve(mc1, mc2)
        self.assertEqual(len(res), 0)
        
    def test_mazeprops10(self):
        mc1 = MazeClause([(("X", (1, 1)), True), (("Y", (1, 1)), False), (("Z", (1, 1)), True)])
        mc2 = MazeClause([(("X", (1, 1)), False), (("Y", (1, 1)), False), (("W", (1, 1)), False)])
        res = MazeClause.resolve(mc1, mc2)
        self.assertEqual(len(res), 1)
        self.assertTrue(MazeClause([(("Y", (1, 1)), False), (("Z", (1, 1)), True), (("W", (1, 1)), False)]) in res)
        
    # Added from the skeleton
    def test_mazeprops11(self):
        mc = MazeClause([(("A", (1, 0)), True), (("A", (0, 0)), False)])
        self.assertTrue(mc.get_prop(("A", (1, 0))))
        self.assertFalse(mc.get_prop(("A", (0, 0))))
        self.assertFalse(mc.is_valid())

    def test_mazeprops12(self):
        mc = MazeClause([(("B", (1, 4)), True), (("B", (1, 4)), True)])
        self.assertTrue(mc.get_prop(("B", (1, 4))))
        self.assertFalse(mc.is_valid())
        self.assertFalse(mc.is_empty())

    def test_mazeprops13(self):
        mc = MazeClause([(("A", (1, 0)), True), (("A", (0, 0)), False), (("B", (1, 4)), True), (("B", (1, 4)), False)])
        self.assertTrue(mc.is_valid())
        self.assertFalse(mc.is_empty())

    def test_mazeprops14(self):
        mc1 = MazeClause([(("A", (1, 0)), True), (("A", (0, 0)), False), (("A", (1, 4)), True), (("A", (2, 0)), False)])
        res = MazeClause.resolve(mc1, mc1)
        self.assertEqual(len(res), 0)

    def test_mazeprops15(self):
        mc1 = MazeClause([(("A", (1, 0)), True), (("A", (0, 0)), False), (("A", (1, 4)), True), (("A", (2, 0)), False)])
        mc2 = MazeClause([(("A", (1, 0)), True), (("A", (0, 0)), False), (("A", (1, 4)), False), (("A", (2, 0)), False)])
        res = MazeClause.resolve(mc1, mc2)
        self.assertEqual(len(res), 1)
        self.assertTrue(MazeClause([(("A", (1, 0)), True), (("A", (0, 0)), False), (("A", (2, 0)), False)]) in res)

    def test_mazeprops16(self):
        mc1 = MazeClause([(("A", (1, 0)), True), (("A", (0, 0)), False), (("A", (1, 4)), True), (("A", (2, 0)), False)])
        mc2 = MazeClause([(("A", (1, 0)), True), (("A", (0, 0)), False), (("A", (1, 4)), False), (("A", (2, 0)), True)])
        res = MazeClause.resolve(mc1, mc2)
        self.assertEqual(len(res), 0)
    
    # -----------------------------------------------------------------------------------------
    # MazeKB Tests
    # -----------------------------------------------------------------------------------------
    
    def test_mazekb1(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("X", (1, 1)), True)])))
        
    def test_mazekb2(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), False)]))
        kb.tell(MazeClause([(("X", (1, 1)), True), (("Y", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("Y", (1, 1)), True)])))
        
    def test_mazekb3(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), False), (("Y", (1, 1)), True)]))
        kb.tell(MazeClause([(("Y", (1, 1)), False), (("Z", (1, 1)), True)]))
        kb.tell(MazeClause([(("W", (1, 1)), True), (("Z", (1, 1)), False)]))
        kb.tell(MazeClause([(("X", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("W", (1, 1)), True)])))
        self.assertFalse(kb.ask(MazeClause([(("Y", (1, 1)), False)])))

    # Added from the skeleton
    def test_mazekb4(self):
        # The Great Forneybot Uprising!
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("M", (1, 1)), False), (("D", (1, 1)), True), (("P", (1, 1)), True)]))
        kb.tell(MazeClause([(("D", (1, 1)), False), (("M", (1, 1)), True)]))
        kb.tell(MazeClause([(("P", (1, 1)), False), (("M", (1, 1)), True)]))
        kb.tell(MazeClause([(("R", (1, 1)), False), (("W", (1, 1)), True), (("S", (1, 1)), True)]))
        kb.tell(MazeClause([(("R", (1, 1)), False), (("D", (1, 1)), True)]))
        kb.tell(MazeClause([(("D", (1, 1)), False), (("R", (1, 1)), True)]))
        kb.tell(MazeClause([(("P", (1, 1)), False), (("F", (1, 1)), True)]))
        kb.tell(MazeClause([(("F", (1, 1)), False), (("P", (1, 1)), True)]))
        kb.tell(MazeClause([(("F", (1, 1)), False), (("S", (1, 1)), False)]))
        kb.tell(MazeClause([(("F", (1, 1)), False), (("W", (1, 1)), False)]))
        kb.tell(MazeClause([(("S", (1, 1)), False), (("W", (1, 1)), False)]))
        kb.tell(MazeClause([(("M", (1, 1)), True)]))
        kb.tell(MazeClause([(("F", (1, 1)), True)]))
        
        # asking alpha = !D ^ P should return True; KB does entail alpha
        kb1 = deepcopy(kb)
        kb1.tell(MazeClause([(("D", (1, 1)), False)]))
        self.assertTrue(kb1.ask(MazeClause([(("P", (1, 1)), True)])))

        kb2 = deepcopy(kb)
        kb2.tell(MazeClause([(("P", (1, 1)), True)]))
        self.assertTrue(kb2.ask(MazeClause([(("D", (1, 1)), False)])))

    def test_mazekb5(self):
        kb = MazeKnowledgeBase()
        # If it is raining, then the sidewalk is wet. !R v S
        kb.tell(MazeClause([(("R", (1, 1)), False), (("S", (1, 1)), True)]))

        # It's raining; KB entails that sidewalk is wet
        kb1 = deepcopy(kb)
        kb1.tell(MazeClause([(("R", (1, 1)), True)]))
        self.assertTrue(kb1.ask(MazeClause([(("S", (1, 1)), True)])))

        # The sidewalk's wet; KB does not entail that it's raining
        kb2 = deepcopy(kb)
        kb2.tell(MazeClause([(("S", (1, 1)), True)]))
        self.assertFalse(kb2.ask(MazeClause([(("R", (1, 1)), True)])))

    def test_mazekb6(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (0, 0)), True), (("Z", (0, 0)), True), (("Y", (0, 0)), True)]))
        kb.tell(MazeClause([(("Z", (0, 0)), False), (("W", (0, 0)), True), (("X", (0, 0)), True)]))
        kb.tell(MazeClause([(("X", (0, 0)), False), (("W", (0, 0)), True)]))
        kb.tell(MazeClause([(("W", (0, 0)), False)]))

        # KB does entail alpha = !X ^ Y
        kb.tell(MazeClause([(("X", (0, 0)), False)]))
        self.assertTrue(kb.ask(MazeClause([(("Y", (0, 0)), True)])))


# -----------------------------------------------------------------------------------------
# MazeAgent Tests
# -----------------------------------------------------------------------------------------

MIN_SCORE = -100
NUM_ITERS = 5

def run_one_maze (maze):
    scores = [max(Environment(maze, tick_length = 0, verbose = False).start_mission(), -100) for iter in range(NUM_ITERS)]
    return sum(scores) / NUM_ITERS

def run_mazes (mazes, grading_threshold=-100, title="Maze Pitfall Tests"):
    """
    Runs and scores a list of mazes based on an average agent performance on each
    :param mazes: list of mazes to run the agent upon
    :param grading_threshold: threshold to do better-than in order to have "passed"
           the given maze
    :param title: string indicating the title of the given tests to print out 
    """
    # Twist: duplicate the input mazes... in reverse!
    print("----------------------------")
    print("[!] Tests Running: " + title)
    new_mazes = deepcopy(mazes)
    for m in new_mazes:
        m.reverse()
    mazes.extend(new_mazes)
    total, attempted, passed = 0, 0, 0
    for maze in mazes:
        attempted += 1
        maze_avg = 0
        try:
            maze_avg = run_one_maze(maze)
        except:
            print(" [X] Error on maze")
            maze_avg = -100
        total += maze_avg
        if maze_avg > grading_threshold:
            passed += 1
        else:
            print("  [X] Failed maze with score of " + str(maze_avg))
            print("\n".join(maze))
    print("  Total Score: " + str(total))
    print("----------------------------")
    return (total, attempted, passed)

def report_results (results_list, title):
    total = sum([x[0] for x in results_list])
    attempted = sum([x[1] for x in results_list])
    passed = sum([x[2] for x in results_list])
    print("=============================")
    print("[!] Final Report on " + title)
    print("  [>] Passed: " + str(passed) + " / " + str(attempted))
    print("  [>] Total score: " + str(total))
    print("=============================")

    
def grading_mazes():
    mazes = [
        # Easy difficulty: Score > -20
        ["XXXXXX",
         "X...GX",
         "X..PPX",
         "X....X",
         "X..P.X",
         "X@...X",
         "XXXXXX"],
             
        ["XXXXXX",
         "XG...X",
         "X..PPX",
         "X....X",
         "X..P.X",
         "X...@X",
         "XXXXXX"],
        
        ["XXXX",
         "XG.X",
         "XP.X",
         "XP.X",
         "X..X",
         "X@.X",
         "XXXX"],
        
        # Medium difficulty: Score > -30
        ["XXXXXXXXX",
         "X..PGP..X",
         "X.......X",
         "X..P.P..X",
         "X.......X",
         "X..@....X",
         "XXXXXXXXX"],
             
        ["XXXXXXXXX",
         "X..P.P.GX",
         "X@......X",
         "X..P.P..X",
         "X.......X",
         "X.......X",
         "XXXXXXXXX"],
        
        ["XXXXXXXXX",
         "X..P.P.GX",
         "X@..P...X",
         "X..P.P..X",
         "X.......X",
         "X.......X",
         "XXXXXXXXX"],
        
        ["XXXXXXXXX",
         "X......GX",
         "X.......X",
         "X.PPPPPPX",
         "X.......X",
         "X......@X",
         "XXXXXXXXX"],
             
        ["XXXXXXXXX",
         "X.......X",
         "X..PGP..X",
         "X...P...X",
         "X.......X",
         "X...P..@X",
         "XXXXXXXXX"],
        
        # Hard difficulty: Score > -40
        ["XXXXXXXXX",
         "XG.P....X",
         "X.......X",
         "X.PP.PP.X",
         "XP.....PX",
         "X...@...X",
         "XXXXXXXXX"],
             
        ["XXXXXXXXX",
         "X...G...X",
         "X.......X",
         "X.P.P.P.X",
         "XP.....PX",
         "X...@...X",
         "XXXXXXXXX"],
             
        ["XXXXXXXXX",
         "XG......X",
         "X....P..X",
         "X.PPPPP.X",
         "XP......X",
         "X...@...X",
         "XXXXXXXXX"],
             
        ["XXXXXXXXX",
         "XP..G..PX",
         "X...P.PPX",
         "XP.....PX",
         "X.......X",
         "XP.P.P.PX",
         "X...@...X",
         "XXXXXXXXX"]
    ]
    
    all_runs = []
    all_runs.append(run_mazes(mazes[0:3], -20, "Easy Grading Mazes"))
    all_runs.append(run_mazes(mazes[3:8], -30, "Medium Grading Mazes"))
    all_runs.append(run_mazes(mazes[8: ], -40, "Hard Grading Mazes"))
    return all_runs

def competition_mazes():
    mazes = [
        # Competition mazes go here
    ]
    
    return [run_mazes(mazes, -100, "Competition Mazes")]
        
        
if __name__ == "__main__":
    report_results(grading_mazes(), "Grading Mazes")
    report_results(competition_mazes(), "Competition Mazes")
    timeout_decorator.timeout(2, use_signals=False)(unittest.main)()