# Results
This graph shows the results for the quantity vs diversity experiment. The experiment was conducted to investigate the effect of the number of syntactic tokens for the generation of semantic tokens while also considering the amount of beams during the generation.

<style>
  .grid-container {
    display: grid;
    gap: 1rem;
    grid-template-columns: repeat(2, 1fr);
  }

  @media (min-width: 1600px) {
    .grid-container {
      grid-template-columns: repeat(4, 1fr);
    }
  }

  .grid-item {
    margin: 10px;
  }

  .grid-item img {
    width: 100%;
    height: auto;
  }
</style>

<div class="grid-container">
    <div class="grid-item">
        <img src="./animations/3d_graph_animated_25s.gif" alt="All results in one graph">
    </div>
    <div class="grid-item">
        <img src="./animations/3d_graph_animated_no_ratio_25s.gif" alt="All results in one graph">
    </div>
    <div class="grid-item">
        <img src="./animations/3d_graph_animated_no_abs_25s.gif" alt="All results in one graph">
    </div>
    <div class="grid-item">
        <img src="./animations/3d_graph_animated_no_abs_no_ratio_25s.gif" alt="All results in one graph">
    </div>
</div>

If we plot the beams as individual lines at `amount_syntactic_tokens = 1` in a 2d graph and mark the highest values with x's, we get the following result:
<div style="display: flex; justify-content: center; ">
    <img src="./2d_plots/beams_to_semantic_tokens.png" alt="All results for 4 generated syntactic tokens" style="max-width: 800px;">
</div>
We can see, that 4 syntactic tokens seem to be the best choice for the generation. The linearity of growth of semantic tokens is clearly visualized when we gather all the amount of semantic tokens at the best amount of syntactic tokens. If we now plot the amount of semantic tokens (y-axis) against the amount of beams (x-axis), we get the following graph:
<div style="display: flex; justify-content: center; ">
    <img src="./2d_plots/growth_semantic_tokens_with_beams.png" alt="All results for 4 generated syntactic tokens" style="max-width: 800px;">
</div>