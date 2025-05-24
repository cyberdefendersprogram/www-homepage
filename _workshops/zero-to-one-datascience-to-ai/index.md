---
layout: blog 
title: Zero To One DataScience To Machine Learning
order: 000
date: 2025-05-23
slug: zero-to-one-datascience-to-machinelearning
workshop_name: zero-to-one-datascience-to-machinelearning
tags: [python, data-science, specific-topic]
---
{% assign filtered_items = site.workshops | where_exp: "item", "item.order != 0 " %}
{% assign sorted_filtered_items = filtered_items | sort: "order" %}

{% if sorted_filtered_items.size > 0 %}
<h1>Workshop {{page.title}}</h1>
  <ul class="collection-list">
    {% for item in sorted_filtered_items %}
      <li class="collection-item">
      <h3> {{item.order}} 
          <a href="{{ item.url | relative_url }}">
            {{ item.title | escape }}
          </a>
        </h3>
      </li>
    {% endfor %}
  </ul>
{% else %}
  <p>There are no items in this collection yet.</p>
{% endif %}
