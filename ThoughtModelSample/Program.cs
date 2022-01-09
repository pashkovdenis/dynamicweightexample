using ThoughtModelSample;
Console.WriteLine("Thought container example");





var thought = new Thought
{
    Decisions = new List<Decision>
        {
            new Decision
            {
                Answer = "True",
                Id =1,
                Symbols = new List<Symbol>{new Symbol { Word = "1"}, new Symbol { Word = "1" } }
            },
            new Decision
            {
                Answer = "False",
                Id = 2,
                Symbols = new List<Symbol>{new Symbol { Word = "1"}, new Symbol { Word = "0" } }
            },
            new Decision
            {
                Answer = "True",
                Id = 3,
                Symbols = new List<Symbol>{new Symbol { Word = "0"}, new Symbol { Word = "0" } }
            },
            new Decision
            {
                Answer = "False",
                Id = 4,
                Symbols = new List<Symbol>{new Symbol { Word = "0"}, new Symbol { Word = "1" } }
            },
            new Decision
            {
                Answer = "Hello",
                Id = 5,
                Symbols = new List<Symbol>{new Symbol { Word = "Hi"}, new Symbol { Word = "There" } }
            },
        }
};

thought.Decisions.ForEach(x => x.ResetWeights());


thought.Reinforce(new[] { "0", "0" }, 3);
thought.Reinforce(new[] { "1", "0" }, 2);
thought.Reinforce(new[] { "0", "1" }, 4);
thought.Reinforce(new[] { "1", "1" }, 1);
thought.Reinforce(new[] { "Hi", "There" }, 5);

Validate(thought.GetAnswers(new[] { "1", "1" }).First().Answer, "True");
Validate(thought.GetAnswers(new[] { "1", "0" }).First().Answer, "False");
Validate(thought.GetAnswers(new[] { "0", "1" }).First().Answer, "False");
Validate(thought.GetAnswers(new[] { "0", "0" }).First().Answer, "True");
Validate(thought.GetAnswers(new[] { "Hi", "There" }).First().Answer, "Hello");




static void Validate(string result, string expected)
{
    Console.WriteLine(result == expected ? "Valid" : "False");


    if (result != expected)
    {
        Console.ReadKey();
    }

}

// Ready for action over here 
Console.WriteLine("Ready for action");



// Region of the custom structures.
public class Symbol
{
    public string Word { get; init; }

    public double Weight { get; set; }

}

/// <summary>
/// Represent a single decision within one thought
/// this decision can act as one of possible choices depending on stimulus
/// </summary>
///

public class Decision
{
    public long Id { get; init; }
    public List<Symbol> Symbols { get; init; } = new();

    public double Bias { get; set; }
    public double Delta { get; set; }
    public string Answer { get; set; }

    // ResetWeights
    public void ResetWeights()
    {
        foreach (var symbol in Symbols)
        {
            symbol.Weight = MathUtils.CalculateRandomDistribution(10);
        }

        Bias = MathUtils.CalculateRandomDistribution(Symbols.Count);
    }

    public void UpdateWeights()
    {
        Bias += Delta;

        foreach (var symbol in Symbols)
        {
            symbol.Weight += Delta;
        }

    }


    // TODO: Pass normalization action here ?
    public double Stimulate(string[] words)
    {
        var matchedSymbols = Symbols.Where(s => words.Contains(s.Word));

        var score = matchedSymbols.Sum(x => x.Weight) + Bias;

        return MathUtils.Sigmoid(score);
    }


    public override string ToString() => Answer;
}


/// <summary>
/// Logical grouping of decisions
/// cluster of decisions that represents a thought
/// </summary>
public class Thought
{
    public List<Decision> Decisions { get; init; } = new();

    public Output[] GetAnswers(string[] stimulus) => Decisions
        .Select(x => new Output(x.Stimulate(stimulus), x.Answer, x.Id))
        .OrderByDescending(x => x.Score)
        .ToArray();

    /// <summary>
    /// Reinforce single thought
    /// Updates symbols weights in order to get predicted output for a single thought
    /// basically it is a kind of back-propagation.
    /// </summary>
    /// <param name="stimulus"></param>
    /// <param name="expected"></param>
    public void Reinforce(string[] stimulus, long expectedDecisionId)
    {
        var repeatTimes = Decisions.Count * 1300;

        for (var i = 0; i < repeatTimes; i++)
        {
            var currentAnswers = GetAnswers(stimulus);
            var answer = currentAnswers.First();

            var targetDecision = Decisions.First(d => d.Id == expectedDecisionId);
            var targetAnswer = currentAnswers.First(a => a.DecisionId == targetDecision.Id);

            targetDecision.Delta = MathUtils.Derivative(targetAnswer.Score) * (targetAnswer.Score * answer.Score);
            targetDecision.UpdateWeights();

            var otherDecisions = currentAnswers.Where(x => x.DecisionId != expectedDecisionId)
                .Select(x => x.DecisionId)
                .ToArray();

            var updateDelta = MathUtils.Derivative(answer.Score) * targetDecision.Delta * targetAnswer.Score;

            foreach (var dec in otherDecisions)
            {
                var currentDecision = Decisions.First(d => d.Id == dec);

                currentDecision.Delta = updateDelta;
                currentDecision.UpdateWeights();
            }
        }
    }
}

/// <summary>
/// Represent a thought's output structure 
/// </summary>
/// <param name="Score"></param>
/// <param name="Answer"></param>
public record struct Output(double Score, string Answer, long DecisionId)
{
    public override string ToString() => $"Id: {DecisionId} ; {Answer}({Score})";
}




