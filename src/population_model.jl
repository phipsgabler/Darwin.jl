export assessfitness!,
    PopulationModel

mutable struct PopulationModel{G, F<:AbstractFitness{>:G}} <: AbstractPopulationModel{G}
    population::Population{G}
    fitness::F
    fittest::Individual{G}
    
    PopulationModel(population::Population{G}, fitness) where {G} =
        new{G, typeof(fitness)}(population, fitness)
end


assessfitness!(model::PopulationModel) = model.fittest = maximumby(i -> assess!(i, model.fitness),
                                                                   model.population)
