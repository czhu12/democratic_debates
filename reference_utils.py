import matplotlib.pyplot as plt
import networkx as nx

class PeopleGraph:
    def __init__(self, people_data):
        self.people = [Person(**p) for p in people_data]
        self.edges = []

    def find_person(self, string_name):
        matches = []
        for p in self.people:
            match_score = p.match_score(string_name)
            if match_score > 0:
                matches.append({'person': p, 'score': match_score})

        # If there is a tie, lets create a new person
        matches = sorted(matches, key=lambda m: -m['score'])
        if len(matches) == 0 or (len(matches) > 1 and matches[0]['score'] == matches[1]['score']):
            p = Person(name=string_name, keywords=string_name)
            self.people.append(p)
            return p
        else:
            p = matches[0]['person']
            p.add_to_keywords(string_name)
            return p

    def add_reference(self, person_a_string, person_b_string):
        person_a = self.find_person(person_a_string)
        person_b = self.find_person(person_b_string)

        found_edge = [e for e in self.edges if e['from'] == person_a and e['to'] == person_b]
        if len(found_edge) > 0:
            found_edge[0]['count'] += 1
        else:
            self.edges.append({'from': person_a, 'to': person_b, 'count': 1})

    def draw(self, num_edges=10):
        print("Drawing {} edges".format(num_edges))
        G = nx.DiGraph(directed=True)
        edges_to_draw = sorted(self.edges, key=lambda e: -e['count'])[:num_edges]
        max_count = max([e['count'] for e in edges_to_draw])
        for e in edges_to_draw:
            G.add_edge(e['from'], e['to'], weight=e['count'] / max_count)

        edges = [(u, v) for (u, v, d) in G.edges(data=True)]
        pos = nx.spring_layout(G)  # positions for all nodes
        nx.draw_networkx_nodes(G, pos, node_size=700)
        nx.draw_networkx_edges(G, pos, edgelist=edges,
                               width=6, arrows=True)
        nx.draw_networkx_labels(G, pos, font_size=9, font_family='sans-serif')

        plt.axis('off')
        plt.show()


class Person:
    def __init__(self, name, keywords):
        keywords = [n for n in keywords.lower().split()]
        self.keyword_counts = dict(zip(keywords, [1] * len(keywords)))
        self.name = name

    def __str__(self):
        return self.name

    def contains(self, key):
        return key in self.keyword_counts

    def add_to_keywords(self, string_name):
        for keyword in set(string_name.lower().split(' ')):
            if keyword in self.keyword_counts:
                self.keyword_counts[keyword] += 1

    def match_score(self, string_name):
        match_keywords = set(string_name.lower().split(' '))
        total = 0
        for match_keyword in match_keywords:
            total += self.keyword_counts[match_keyword] if match_keyword in self.keyword_counts else 0
        return total

graph = PeopleGraph([
    {"name": "Joe Biden", "keywords": 'joe biden vice president j biden'},
    {"name": "Bernie Sanders", "keywords": 'Bernie Sanders'},
    {"name": "Pete Buttigieg", "keywords": 'Pete Buttigieg Mayor Buttigieg'},
])

graph.add_reference('vermont senator bernie sanders', 'Mayor pete')
assert graph.edges[0]['from'].contains('sanders')
assert graph.edges[0]['to'].contains('pete')
assert len(graph.people) == 3
graph.add_reference('vermont senator bernie sanders', 'Mayor pete')
assert len(graph.edges) == 1
assert graph.edges[0]['count'] == 2

graph.add_reference('pete', 'donald trump')

assert graph.edges[1]['from'].contains('pete')
assert graph.edges[1]['to'].contains('trump')
assert len(graph.people) == 4
