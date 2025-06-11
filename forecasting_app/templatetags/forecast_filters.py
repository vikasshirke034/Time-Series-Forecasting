# from django import template

# register = template.Library()

# @register.filter
# def get_item(dictionary, key):
#     return dictionary.get(key)

# @register.filter
# def items(dictionary):
#     return dictionary.items()

# @register.filter
# def addclass(field, css_class):
#     return field.as_widget(attrs={"class": css_class})

from django import template

# register = template.Library()

# @register.filter
# def get_item(dictionary, key):
#     if dictionary is None:
#         return None
#     if isinstance(dictionary, dict):
#         return dictionary.get(key)
#     return getattr(dictionary, key, None)


#     from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    if hasattr(dictionary, 'get'):
        return dictionary.get(key, '')
    return getattr(dictionary, key, '')