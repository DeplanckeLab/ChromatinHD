ChromatinHD analyzes single-cell ATAC+RNA data using the raw fragments as input, by automatically adapting the scale at which
relevant chromatin changes on a per-position, per-cell, and per-gene basis.

Currently, the following models are supported:

<style>
.cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    grid-gap: 20px;
}

.card {
    display: grid;
    grid-template-rows: max-content 200px 1fr;
    background: rgb(236, 236, 236);
    padding:10px;
    box-shadow: rgba(0, 0, 0, 0.2) 0px 2px 4px, rgba(0, 0, 0, 0.15) 0px 7px 13px -3px, rgba(0, 0, 0, 0.1) 0px -3px 0px inset;
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

.card img {
    object-fit: cover;
    width: 100%;
    height: 100%;
}
</style>


<div class="cards">
    <a href="quickstart/2_pred">
    <article class="card">
        <header>
            <h2>Pred</h2>
        </header>    
        <!-- <img src="balloons.jpg" alt="Hot air balloons"> -->
        <div class="content">
            <p> To learn where and how accessibility is predictive for gene expression </p>
        </div>
            
    </article>
    </a>
            <a href="quickstart/3_diff">
     <article class="card">
        <header>
            <h2>Diff</h2>
        </header>    
        <!-- <img src="balloons2.jpg" alt="Hot air balloons"> -->
        <div class="content">
            <p>To understand the differences in accessibilty between cell types/states</p>
        </div>
    </article>
    </a>
</div>