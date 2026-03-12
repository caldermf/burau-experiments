#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int positive_mod(int x,int p)
{
  x %= p;
  if(x < 0)
    x += p;
  return(x);
}

int pairing_mod(int a,int b,int c,int d,int e,int q,int p)
{
  int bl,start,cl,end, el,er;
  int suma,sumb,sumc,sumd,sume;
  int x;
  int poly = 0;
  int monomial = 1;
  int q_mod = positive_mod(q,p);
  int q4 = positive_mod(q_mod*q_mod,p);
  q4 = positive_mod(q4*q4,p);

  bl = 2*a;
  start = bl+b;
  cl = start+b+1;
  end = cl+c;
  el = 2*d;
  er = el+e;

  suma = bl-1;
  sumb = 2*start;
  sumc = 2*end;
  sumd = el-1;
  sume = er + er-1;

  x = start;

  while(1)
  {
    if(x < el)
    {
      if(x < d)
      {
        monomial = positive_mod(monomial*q_mod,p);
        poly = positive_mod(poly + monomial,p);
      }
      else
      {
        poly = positive_mod(poly - monomial,p);
        poly = positive_mod(poly*q_mod,p);
      }
      x = sumd - x;
    }
    else
    {
      if(x < er)
      {
        poly = positive_mod(poly - monomial,p);
        monomial = positive_mod(monomial*q_mod,p);
      }
      else
      {
        poly = positive_mod(poly*q_mod,p);
        poly = positive_mod(poly + monomial,p);
      }
      x = sume - x;
    }

    if(x < cl)
    {
      if(x < bl)
      {
        if(x < a)
          poly = positive_mod(poly*q_mod,p);
        else
          monomial = positive_mod(monomial*q_mod,p);
        x = suma-x;
      }
      else
      {
        if(x < start)
          monomial = positive_mod(monomial*q4,p);
        else
          poly = positive_mod(poly*q4,p);
        x = sumb-x;
      }
    }
    else
    {
      if(x < end)
        poly = positive_mod(poly*q_mod,p);
      else if(x > end)
        monomial = positive_mod(monomial*q_mod,p);
      else
        return(poly);
      x = sumc-x;
    }
  }
}

int single_whisker(int a,int b,int c,int d,int e)
{
  int bl,start,cl,end, el,er;
  int suma,sumb,sumc,sumd,sume;
  int x;
  int togo = d+e;

  bl = 2*a;
  start = bl+b;
  cl = start+b+1;
  end = cl+c;
  el = 2*d;
  er = el+e;

  suma = bl-1;
  sumb = 2*start;
  sumc = 2*end;
  sumd = el-1;
  sume = er + er-1;

  x = start;

  while(1)
  {
    if(x < el)
      x = sumd - x;
    else
      x = sume - x;
    togo--;

    if(x < cl)
    {
      if(x < bl)
        x = suma-x;
      else
        x = sumb-x;
    }
    else
    {
      if(x == end)
        return(!togo);
      x = sumc-x;
    }
  }
}

void ensure_capacity(int** coeffs,int* cap,int needed)
{
  int old_cap;
  int* new_coeffs;

  if(needed <= *cap)
    return;

  old_cap = *cap;
  while(*cap < needed)
    *cap *= 2;

  new_coeffs = calloc((size_t)(*cap),sizeof(int));
  memcpy(new_coeffs,*coeffs,(size_t)old_cap*sizeof(int));
  free(*coeffs);
  *coeffs = new_coeffs;
}

void shift_poly(int** coeffs,int* len,int* cap,int amount)
{
  int i;

  if(amount == 0 || *len == 0)
    return;

  ensure_capacity(coeffs,cap,*len + amount);
  for(i = *len - 1; i >= 0; i--)
    (*coeffs)[i + amount] = (*coeffs)[i];
  for(i = 0; i < amount; i++)
    (*coeffs)[i] = 0;
  *len += amount;
}

void add_monomial(int** coeffs,int* len,int* cap,int exp,int delta,int p)
{
  int needed = exp + 1;

  ensure_capacity(coeffs,cap,needed);
  while(*len < needed)
  {
    (*coeffs)[*len] = 0;
    (*len)++;
  }
  (*coeffs)[exp] = positive_mod((*coeffs)[exp] + delta,p);
}

int exact_zero_mod_p(int a,int b,int c,int d,int e,int p)
{
  int bl,start,cl,end, el,er;
  int suma,sumb,sumc,sumd,sume;
  int x;
  int mon_exp = 0;
  int len = 0;
  int cap = 64;
  int* coeffs = calloc((size_t)cap,sizeof(int));
  int i;

  bl = 2*a;
  start = bl+b;
  cl = start+b+1;
  end = cl+c;
  el = 2*d;
  er = el+e;

  suma = bl-1;
  sumb = 2*start;
  sumc = 2*end;
  sumd = el-1;
  sume = er + er-1;

  x = start;

  while(1)
  {
    if(x < el)
    {
      if(x < d)
      {
        mon_exp++;
        add_monomial(&coeffs,&len,&cap,mon_exp,1,p);
      }
      else
      {
        add_monomial(&coeffs,&len,&cap,mon_exp,-1,p);
        shift_poly(&coeffs,&len,&cap,1);
      }
      x = sumd - x;
    }
    else
    {
      if(x < er)
      {
        add_monomial(&coeffs,&len,&cap,mon_exp,-1,p);
        mon_exp++;
      }
      else
      {
        shift_poly(&coeffs,&len,&cap,1);
        add_monomial(&coeffs,&len,&cap,mon_exp,1,p);
      }
      x = sume - x;
    }

    if(x < cl)
    {
      if(x < bl)
      {
        if(x < a)
          shift_poly(&coeffs,&len,&cap,1);
        else
          mon_exp++;
        x = suma-x;
      }
      else
      {
        if(x < start)
          mon_exp += 4;
        else
          shift_poly(&coeffs,&len,&cap,4);
        x = sumb-x;
      }
    }
    else
    {
      if(x < end)
        shift_poly(&coeffs,&len,&cap,1);
      else if(x > end)
        mon_exp++;
      else
      {
        for(i = 0; i < len; i++)
          if(coeffs[i] != 0)
          {
            free(coeffs);
            return(0);
          }
        free(coeffs);
        return(1);
      }
      x = sumc-x;
    }
  }
}

int passes_field_filter(int a,int b,int c,int d,int e,int p)
{
  int q;
  for(q = 0; q < p; q++)
    if(pairing_mod(a,b,c,d,e,q,p))
      return(0);
  return(1);
}

int main(void)
{
  int p,a,b,c,d,e;

  while(scanf("%d %d %d %d %d %d",&p,&a,&b,&c,&d,&e) == 6)
  {
    int single = single_whisker(a,b,c,d,e);
    int field = 0;
    int exact = 0;
    if(single)
    {
      field = passes_field_filter(a,b,c,d,e,p);
      exact = exact_zero_mod_p(a,b,c,d,e,p);
    }
    printf("%d %d %d\n",single,field,exact);
  }

  return(0);
}
