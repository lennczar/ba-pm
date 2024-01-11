```
function EvalutateSplits(L, a, h) {

    attribute_values <- sort(F^a_L, ascending)
        
    L_1 <- {}
    L_2 <- ^L

    results <- []

    for pivot in drop_first(attribute_values) {

        affected <- L.filter(lambda(t){t^a < pivot})

        L <- L \ affected

        for t^n in ^affected {

            m <- occ(t, L_1)
            L_1 <- (L_1 \ { t^m }) cup { t^(m+n) }

            m <- occ(t, L_2)
            L_2 <- (L_2 \ { t^m }) cup { t^(m-n) }

        }

        results.push(h(L_1, L_2))

    }

}
```

```
function occ(t, L) {
    if t^n in L {
        return n
    } else {
        return 0
    }
}
```

# pareto stuff for selection of s? DM? just use max? can maybe just always use pareto?

```py
attribute_values = pm4py.get_trace_attribute_values(df, attribute_key)
sorted_attribute_values = sorted([ float(k) for k in attribute_values.keys() ])

# group df by case ids
grouped = df.filter([case_id_key, case_attribute_key]).groupby(case_id_key, sort=False)
df_gte_light = grouped.first() # initial gte group

# define absolute languages (counting the occurrence of cases)
# devide absoulte numbers by total number of cases later(!), otherwise every values has to be updated
language_lt = {}
language_gte = pm4py.stats.get_variants(df)

lt_sizes = [0]
gte_sizes = [len(grouped.first().index)]

uemscs = []

ginis = []
g1s = []
g2s = []
n1s = []
n2s = []

for idx, pivot in enumerate(sorted_attribute_values[1:]):

    # find affected cases
    df_affected_light = df_gte_light[df_gte_light[case_attribute_key] < pivot]

    # remove affected cases from gte group
    df_gte_light = df_gte_light.drop(df_affected_light.index)

    # translate into full df
    df_affected = pd.concat([ df.loc[grouped.indices[i]] for i in df_affected_light.index ])

    # update absoulte languages
    for tr, n in pm4py.stats.get_variants(df_affected).items():
        language_lt[tr] = language_lt.setdefault(tr, 0) + n
        language_gte[tr] = language_gte[tr] - n

    language_lt_size = sum(language_lt.values(), 0.0)
    language_gte_size = sum(language_gte.values(), 0.0)

    stochastic_language_lt = { k: v / language_lt_size for k, v in language_lt.items() }
    stochastic_language_gte = { k: v / language_gte_size for k, v in language_gte.items() }

    uemsc = uEMSC(stochastic_language_lt, stochastic_language_gte)

    n1 = language_lt_size
    n2 = language_gte_size
    n = n1 + n2
    g1 = (1 - sum([ pr**2 for pr in stochastic_language_lt.values() ])) 
    g2 = (1 - sum([ pr**2 for pr in stochastic_language_gte.values() ]))
    gini = (n1/n) * g1 + (n2/n) * g2

    print(f'[{idx}/{len(attribute_values) - 1}]: (uEMSC) {uemsc} | (Gini) {gini}')
    
    # collect data to plot graph
    uemscs.append(uemsc)
    ginis.append(gini)
    n1s.append(n1/n)
    n2s.append(n2/n)
    g1s.append(g1)
    g2s.append(g2)
    lt_sizes.append(language_lt_size)
    gte_sizes.append(language_gte_size)
```