export findfittest!,
    findfittest,
    PopulationModel

mutable struct PopulationModel{G, F<:AbstractFitness{>:G}} <: AbstractEvolutionaryModel
    population::Population{G}
    fitness::F
    fittest::Individual{G}
    
    PopulationModel(population::Population{G}, fitness) where {G} =
        new{G, typeof(fitness)}(population, fitness)
end


findfittest!(model::PopulationModel) = model.fittest = maximumby(i -> assess!(i, model.fitness),
                                                                 model.population)
findfittest(model::PopulationModel) = model.fittest
