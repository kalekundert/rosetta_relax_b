nextflow.enable.dsl = 2

params.n = 50
params.out_dir = 'results'
params.dry_run = false

params.opt = [:]
params.opt.config = ''
params.opt.prior_results = ''

process optimize_cst {
    cpus 1
    memory '2GB'
    time params.dry_run ? '30m' : '96h'
    publishDir params.out_dir, saveAs: {file(it).getName()}, mode: 'link'

    input:
        path 'unrelaxed.pdb'
    output:
        path 'out/optimize_cst.json', emit: results
        path 'out/gaussian_process_model.svg', emit: svg

    """
    optimize_cst.py \
        unrelaxed.pdb \
        ${c = params.opt.config ? "-c $c": ""} \
        ${i = params.opt.prior_results ? "-i $i" : ""} \
        ${params.dry_run ? "-d" : ""} \

    ln -s results.json out/optimize_cst.json
    """
}

process relax {
    cpus 1
    memory '2GB'
    time params.dry_run ? '30m' : '24h'

    input:
        path 'unrelaxed.pdb'
        path 'optimize_cst.json'
        val n
    output:
        path 'relaxed.pdb'

    """
    # Manually specify the input PDB path, because otherwise the script will 
    # try to get it from the JSON input, and that path may not exist anymore 
    # (since it's associated with a different process).
    relax_b.py unrelaxed.pdb -i optimize_cst.json
    """
}

process pick_best_model {
    cpus 1
    memory '1GB'
    time '30m'
    publishDir params.out_dir, pattern: 'best_model.pdb', mode: 'link'

    input:
        path 'models/*.pdb'

    output:
        path 'best_model.pdb'

    """
    pick_best_model.py models
    """
}

workflow {
    unrelaxed = channel.value(file(params.pdb))
    n = channel.of(1..params.n).take(params.dry_run ? 2 : -1)

    opt = optimize_cst(unrelaxed)
    relax(unrelaxed, opt.results, n) | collect | pick_best_model
}
