ChromatinHD analyzes single-cell ATAC+RNA data using the raw fragments as input, by automatically adapting the scale at which
relevant chromatin changes on a per-position, per-cell, and per-gene basis.

Currently, the following models are supported:

<style>
.cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    grid-gap: 20px;
}

.card {
    display: grid;
    grid-template-rows: max-content 250px 1fr;
    background: rgb(255, 255, 255);
    padding:10px;
    box-shadow: rgba(0, 0, 0, 0.2) 0px 2px 4px, rgba(0, 0, 0, 0.15) 0px 7px 13px -3px, rgba(0, 0, 0, 0.1) 0px -3px 0px inset;
    transition: box-shadow 0.1s ease-out 100ms;
    color:#333;
}

.card:hover{
    box-shadow: rgba(0, 0, 0, 0.4) 0px 2px 4px, rgba(0, 0, 0, 0.3) 0px 7px 13px -3px, rgba(0, 0, 0, 0.2) 0px -3px 0px inset;
}

.card h2{
    text-align:center;
}

.card .content{
    margin:10px
}

.card.disabled {
    opacity:0.5;
}

.card img {
  display: block;
  margin-left: auto;
  margin-right: auto;
}

.card h2 {
    font-style:italic;
}
</style>

<div class="cards">
    <a href="quickstart/2_pred">
    <article class="card">
        <header>
            <h2>Pred</h2>
        </header>
        <div class="content">
            <p> To learn where and how accessibility is predictive for gene expression </p>
        <img src="static/models/pred/1x/logo.png" alt="ChromatinHD-pred" style="scale:0.7">
        </div>

    </article>
    </a>
            <a href="quickstart/3_diff">
     <article class="card">
        <header>
            <h2>Diff</h2>
        </header>    
        <div class="content">
            <p>To understand the differences in accessibilty between cell types/states</p>
        <img src="static/models/diff/1x/logo.png" alt="ChromatinHD-diff" style="scale:0.7">
        </div>
    </article>
    </a>
    <article class="card disabled">
        <header>
            <h2>Time</h2>
        </header>    
        <div class="content">
            <p>To learn where and how accessibility is predictive over (pseudo)time</p>
        <img src="static/models/time/logo.png" alt="ChromatinHD-time" style="scale:0.7">
        </div>
    </article>
    <article class="card disabled">
        <header>
            <h2>Dime</h2>
        </header>    
        <div class="content">
            <p>To learn the differences in accessibility over (pseudo)time</p>
        <img src="static/models/dime/logo.png" alt="ChromatinHD-dime" style="scale:0.7">
        </div>
    </article>
</div>
