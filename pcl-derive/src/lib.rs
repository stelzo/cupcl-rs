extern crate proc_macro;

use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(PointXYZI)]
pub fn pointxyzi_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    // compiler error if something fails
    // get type of x, y, z fields
    let fields = match input.data {
        syn::Data::Struct(ref data) => match data.fields {
            syn::Fields::Named(ref fields) => fields,
            _ => unimplemented!(),
        },
        _ => unimplemented!(),
    };

    let mut x_ty = None;
    let mut y_ty = None;
    let mut z_ty = None;
    let mut i_ty = None;

    let mut intensity_field_name = None;

    for field in fields.named.iter() {
        let ident = field.ident.as_ref();
        match ident {
            None => unimplemented!(),
            Some(ident) => {
                if ident == "x" {
                    x_ty = Some(&field.ty);
                } else if ident == "y" {
                    y_ty = Some(&field.ty);
                } else if ident == "z" {
                    z_ty = Some(&field.ty);
                } else if ident == "i" {
                    i_ty = Some(&field.ty);
                    intensity_field_name = Some(ident);
                } else if ident == "intensity" {
                    if i_ty.is_some() {
                        return syn::Error::new_spanned(name, "Both i and intensity fields are present in the struct. Only one is allowed.")
                            .to_compile_error()
                            .into();
                    }
                    i_ty = Some(&field.ty);
                    intensity_field_name = Some(ident);
                }
            }
        }
    }

    if x_ty.is_none() || y_ty.is_none() || z_ty.is_none() || i_ty.is_none() {
        return syn::Error::new_spanned(
            name,
            "All fields (x, y, z, [i|intensity]) must be present in the struct",
        )
        .to_compile_error()
        .into();
    }

    // all must be the same type
    if x_ty.to_token_stream().to_string() != y_ty.to_token_stream().to_string()
        || x_ty.to_token_stream().to_string() != z_ty.to_token_stream().to_string()
    {
        return syn::Error::new_spanned(name, "All fields (x, y, z, i) must be of the same type")
            .to_compile_error()
            .into();
    }

    let coord_type = x_ty.expect("x_ty is None");
    let intensity_field_name = intensity_field_name.expect("intensity_field_name is None");

    let expanded = quote! {

        impl Zero for #name {
            fn zero() -> Self {
                Self::with_xyzi(#coord_type::zero(), #coord_type::zero(), #coord_type::zero(), #coord_type::zero())
            }

            fn is_zero(&self) -> bool {
                self.get_x().is_zero() && self.get_y().is_zero() && self.get_z().is_zero() && self.get_i().is_zero()
            }
        }

        impl std::ops::Add for #name {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self::with_xyzi(self.get_x() + rhs.get_x(), self.get_y() + rhs.get_y(), self.get_z() + rhs.get_z(), self.get_i() + rhs.get_i())
            }
        }

        impl Point<#coord_type> for #name {
            #[inline(always)]
            fn get_x(&self) -> #coord_type {
                self.x
            }

            #[inline(always)]
            fn get_y(&self) -> #coord_type {
                self.y
            }

            #[inline(always)]
            fn get_z(&self) -> #coord_type {
                self.z
            }

            #[inline(always)]
            fn get_i(&self) -> #coord_type {
                self.#intensity_field_name
            }

            #[inline(always)]
            fn with_xyzi(x: #coord_type, y: #coord_type, z: #coord_type, i: #coord_type) -> Self {
                let mut p = Self::default();
                p.set_x(x);
                p.set_y(y);
                p.set_z(z);
                p.set_i(i);
                p
            }

            #[inline(always)]
            fn with_xyzif64(x: f64, y: f64, z: f64, i: f64) -> Self {
                let mut p = Self::default();
                p.set_x(x as #coord_type);
                p.set_y(y as #coord_type);
                p.set_z(z as #coord_type);
                p.set_i(i as #coord_type);
                p
            }

            #[inline(always)]
            fn set_x(&mut self, x: #coord_type) {
                self.x = x;
            }

            #[inline(always)]
            fn set_y(&mut self, y: #coord_type) {
                self.y = y;
            }

            #[inline(always)]
            fn set_z(&mut self, z: #coord_type) {
                self.z = z;
            }

            #[inline(always)]
            fn set_i(&mut self, i: #coord_type) {
                self.#intensity_field_name = i;
            }
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_derive(PointXYZ)]
pub fn pointxyz_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    let fields = match input.data {
        syn::Data::Struct(ref data) => match data.fields {
            syn::Fields::Named(ref fields) => fields,
            _ => unimplemented!(),
        },
        _ => unimplemented!(),
    };

    let mut x_ty = None;
    let mut y_ty = None;
    let mut z_ty = None;

    for field in fields.named.iter() {
        let ident = field.ident.as_ref();
        match ident {
            None => unimplemented!(),
            Some(ident) => {
                if ident == "x" {
                    x_ty = Some(&field.ty);
                } else if ident == "y" {
                    y_ty = Some(&field.ty);
                } else if ident == "z" {
                    z_ty = Some(&field.ty);
                }
            }
        }
    }

    if x_ty.is_none() || y_ty.is_none() || z_ty.is_none() {
        return syn::Error::new_spanned(
            name,
            "All fields (x, y, z, [i|intensity]) must be present in the struct",
        )
        .to_compile_error()
        .into();
    }

    if x_ty.to_token_stream().to_string() != y_ty.to_token_stream().to_string()
        || x_ty.to_token_stream().to_string() != z_ty.to_token_stream().to_string()
    {
        return syn::Error::new_spanned(name, "All fields (x, y, z, i) must be of the same type")
            .to_compile_error()
            .into();
    }

    let coord_type = x_ty.expect("x_ty is None");

    let expanded = quote! {
        impl Zero for #name {
            fn zero() -> Self {
                Self::with_xyzi(#coord_type::zero(), #coord_type::zero(), #coord_type::zero(), #coord_type::zero())
            }

            fn is_zero(&self) -> bool {
                self.get_x().is_zero() && self.get_y().is_zero() && self.get_z().is_zero()
            }
        }

        impl std::ops::Add for #name {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self::with_xyzi(self.get_x() + rhs.get_x(), self.get_y() + rhs.get_y(), self.get_z() + rhs.get_z(), #coord_type::zero())
            }
        }

        impl Point<#coord_type> for #name {
            #[inline(always)]
            fn get_x(&self) -> #coord_type {
                self.x
            }

            #[inline(always)]
            fn get_y(&self) -> #coord_type {
                self.y
            }

            #[inline(always)]
            fn get_z(&self) -> #coord_type {
                self.z
            }

            #[inline(always)]
            fn get_i(&self) -> #coord_type {
                #coord_type::zero()
            }

            #[inline(always)]
            fn with_xyzi(x: #coord_type, y: #coord_type, z: #coord_type, _: #coord_type) -> Self {
                let mut p = Self::default();
                p.set_x(x);
                p.set_y(y);
                p.set_z(z);
                p
            }

            #[inline(always)]
            fn with_xyzif64(x: f64, y: f64, z: f64, i: f64) -> Self {
                let mut p = Self::default();
                p.set_x(x as #coord_type);
                p.set_y(y as #coord_type);
                p.set_z(z as #coord_type);
                p.set_i(i as #coord_type);
                p
            }

            #[inline(always)]
            fn set_x(&mut self, x: #coord_type) {
                self.x = x;
            }

            #[inline(always)]
            fn set_y(&mut self, y: #coord_type) {
                self.y = y;
            }

            #[inline(always)]
            fn set_z(&mut self, z: #coord_type) {
                self.z = z;
            }

            #[inline(always)]
            fn set_i(&mut self, _: #coord_type) {}
        }
    };

    TokenStream::from(expanded)
}
